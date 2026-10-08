import random
import os
import PIL
import time
import torch
import warnings
warnings.filterwarnings("ignore")
import sys

sys.path.append('./')




from transformers import set_seed
from tqdm import tqdm
from transformers import logging
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, DDIMScheduler

import torch.nn as nn
import numpy as np
import utils.feature_utils as fu
import utils.preprocesser_utils as pu
import utils.image_process_utils as ipu

logging.set_verbosity_error()


def set_seed_lib(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    set_seed(seed)

@torch.no_grad()
class RAVE(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.dtype = torch.float

    @torch.no_grad()
    def __init_pipe(self, hf_cn_path, hf_path):
            """
            hf_cn_path：huggingface_controlnet_path,从huggingface下载的controlnet模型，在本地的位置。
            hf_path：huggingface_path,从huggingface下载的stable diffusion模型，在本地的位置。
            从预训练模型在本地的路径上加载pipline，并返回pipline对象pipe。
            与vedio-p2p不同，这里没有给出pipline的代码实现细节。
            """
            # 从预训练模型路径加载ControlNetModel，并将其转移到指定的设备和数据类型上
            controlnet = ControlNetModel.from_pretrained(hf_cn_path, torch_dtype=self.dtype).to(self.device, self.dtype)
            print(hf_path)
            print(hf_cn_path)
            # 从预训练模型路径加载StableDiffusionControlNetImg2ImgPipeline，并设置controlnet参数，然后将其转移到指定的设备和数据类型上
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(hf_path, controlnet=controlnet, torch_dtype=self.dtype).to(self.device, self.dtype) 
            # 启用模型的CPU卸载功能，以优化内存使用
            pipe.enable_model_cpu_offload()
            # 启用Xformers内存高效注意力机制，以提高性能
            pipe.enable_xformers_memory_efficient_attention()
            # 返回初始化后的管道对象
            return pipe
        
    @torch.no_grad()
    def init_models(self, hf_cn_path, hf_path, preprocess_name, model_id=None):
        """
        hf_cn_path：huggingface_controlnet_path,从huggingface下载的controlnet模型，在本地的位置。
        hf_path：huggingface_path,从huggingface下载的stable diffusion模型，在本地的位置。
        preprocess_name:预处理的方式类别（预处理有多种方式）
        model_id:从本地加载的stable_diffusion model的id
        从本地加载pipe模型并初始化各类模型（从pipe中提起出其他子模型）
        """
        # 如果model_id为空或等于"None"，则使用hf_cn_path和hf_path初始化pipe
        if model_id is None or model_id == "None":
            pipe = self.__init_pipe(hf_cn_path, hf_path)  
        # 否则，使用hf_cn_path和model_id初始化pipe
        else:
            pipe = self.__init_pipe(hf_cn_path, model_id)  
        # 设置预处理名称
        self.preprocess_name = preprocess_name
        
        # 从pipe中获取并设置相关属性
        self._prepare_control_image = pipe.prepare_control_image
        self.run_safety_checker = pipe.run_safety_checker
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
    
        # 从pipe中获取并设置vae和unet模型
        self.vae = pipe.vae
        self.unet = pipe.unet    
    
        # 从pipe中获取并设置controlnet和scheduler配置
        self.controlnet = pipe.controlnet
        self.scheduler_config = pipe.scheduler.config        
        
        # 删除pipe对象，释放内存
        del pipe
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        """
        Parameter:
        prompt:只有一条target prompt，没有source prompt。
        negative_prompt:其实就是无条件输入，类似空文本。
        对提示文本和负面提示文本（空文本）进行编码，并返回条件和无条件嵌入向量。
        """
        # 对提示文本进行编码，并限制最大长度为模型的最大长度
        cond_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        # 使用编码后的输入通过文本编码器得到条件嵌入
        cond_embeddings = self.text_encoder(cond_input.input_ids.to(self.device))[0]
    
        # 对负面提示文本进行编码，并限制最大长度为模型的最大长度
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        # 使用编码后的输入通过文本编码器得到无条件嵌入
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
    
        # 返回条件和无条件嵌入
        return cond_embeddings, uncond_embeddings

    @torch.no_grad()
    def prepare_control_image(self, control_pil, width, height):
        """
       Parameter:
       control_pil:由preprocessor提取input vedio得出的特征图谱。
       width、height：指定的高宽
       将传入的control pil格式化处理，封装各类属性。
        """

        control_image = self._prepare_control_image(
        image=control_pil,
        width=width,
        height=height,
        device=self.device,
        dtype=self.controlnet.dtype,
        batch_size=1,
        num_images_per_prompt=1
    )

        return control_image
    
    @torch.no_grad()
    def pred_controlnet_sampling(self, current_sampling_percent, latent_model_input, t, text_embeddings, control_image):
        """
        parameter：
        current_sampling_percent：当前采样进度百分比。
        latent_model_input：模型输入的latent变量（当前时间步的latent）。
        t：当前时间步。
        text_embeddings：文本嵌入向量。
        control_image：控制图像。
        根据当前采样进度百分比，由controlnet模型通过latent_model_input, t, text_embeddings, control_image做噪声预测。
        本质上还是unet做的噪声预测。返回预测出来的噪声（包括有条件的和无条件的预测噪声）。
        """
        # 如果当前采样百分比小于控制网络引导开始或大于控制网络引导结束，则将down_block_res_samples和mid_block_res_sample设置为None
        if (current_sampling_percent < self.controlnet_guidance_start or current_sampling_percent > self.controlnet_guidance_end):
            down_block_res_samples = None
            mid_block_res_sample = None
        else:
            # 否则，调用controlnet方法，传入latent_model_input、t、conditioning_scale、encoder_hidden_states、controlnet_cond和return_dict参数
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                conditioning_scale=self.controlnet_conditioning_scale,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control_image,
                return_dict=False,
            )
        # 使用unet模型进行预测，传入latent_model_input、t、encoder_hidden_states、down_block_additional_residuals和mid_block_additional_residual参数
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings,                    
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample)['sample']
        # 返回噪声预测值
        return noise_pred
    
    
    @torch.no_grad()
    def denoising_step(self, latents, control_image, text_embeddings, t, guidance_scale, current_sampling_percent):
            """
           Parameter:
           latents:潜在噪声（Zt-1）。
           control_image:控制图像。
           text_embeddings:文本嵌入向量。
           t:当前时间步。
           guidance_scale:CFG中的w值。
           current_sampling_percent:当前采样进度百分比。
           根据预测出的噪声，做一步采样操作（带CFG）。
            """
            
            # 将输入的latents和control_image进行复制，以匹配模型的输入要求
        
            latent_model_input = torch.cat([latents] * 2)
            control_image = torch.cat([control_image] * 2)
            # 根据当前时间步t调整模型输入的大小
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
            # 使用预测控制网络对当前的采样百分比、模型输入、时间步和文本嵌入进行预测
            noise_pred = self.pred_controlnet_sampling(current_sampling_percent, latent_model_input, t, text_embeddings, control_image)
    
            # 将预测结果分为无条件部分和有条件部分
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # 根据引导系数计算最终的噪声预测值
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
            # 使用调度器更新latents，并返回更新后的latents
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
            return latents


    @torch.no_grad()
    def preprocess_control_grid(self, image_pil):
        """
        对控制网格进行预处理。该函数接受一个PIL图像，将其转换为一系列帧，然后对每一帧进行像素完美处理。
        最后，将这些处理后的帧重新组合成一个网格，并将其转换为灰度图像。
        
        Parameters:
        image_pil (PIL.Image): 一个待处理的PIL图像（四合一大图）。
        
        Returns:
        PIL.Image: 返回一个经过预处理（preprocess）的控制网格图像（四合一的大numpy）--从image变成了control image。
    
        """
        # 将一个PIL图像转换为一系列帧并放入帧列表，grid_size=[2,2]
        list_of_image_pils = fu.pil_grid_to_frames(image_pil, grid_size=self.grid) # List[C, W, H] -> len = num_frames
        # 对每个帧进行像素完美处理，处理方式由self.preprocess_name指定
        list_of_pils = [pu.pixel_perfect_process(np.array(frame_pil, dtype='uint8'), self.preprocess_name) for frame_pil in list_of_image_pils]
        # 将处理后的帧列表转换为numpy数组
        control_images = np.array(list_of_pils)
        # 从numpy数组创建一个新的网格（四合一的大numpy向量），网格大小由self.grid决定
        control_img = ipu.create_grid_from_numpy(control_images, grid_size=self.grid)
        # 将网格转换为灰度图像
        control_img = PIL.Image.fromarray(control_img).convert("L")
    
        return control_img
    
    @torch.no_grad()
    def shuffle_latents(self, latents, control_image, indices,t):
            """
            对输入的latents和control_image进行随机洗牌。
            该函数首先生成一个随机排列的索引列表，然后根据这个索引列表从latents和control_image中选取对应的元素，
            最后返回洗牌后的latents、indices和control_image。
            
            Parameters:
            latents (torch.Tensor): 待洗牌的latents张量。四合一大图
            control_image (torch.Tensor): 待洗牌的control_image张量。
            indices (list): 用于记录每个元素在原始latents和control_image中的位置的列表。
            
            Returns:
            tuple: 返回一个元组，包含三个元素：洗牌后的latents张量，洗牌后的indices列表，以及洗牌后的control_image张量。
            """    
            # 生成一个随机排列的索引列表  total_frame_number：vedio包含的帧数
            if t%2==0:
                rand_i=list(range(0, 41))
            else: 
                
                standard_list = list(range(0, 41))
                # 2. 获取从2开始到最后的子列表，再连接上从0开始到2之前的子列表
                my_list = standard_list[2:] + standard_list[:2]
                rand_i=my_list
            rand_i = torch.randperm(self.total_frame_number).tolist()
            #rand_i = torch.arange(self.total_frame_number).tolist()
            #print("rand_i=",rand_i)
            
            # 初始化空列表用于存储洗牌后的latents和control_image，indices
            latents_l, controls_l, randx = [], [], []
            # 遍历采样大小范围内的所有元素 sample_size:网格数目
            for j in range(self.sample_size):
                # 根据当前循环次数获取对应的随机索引  grid_frame_number：一个grid包含的frame数
                rand_indices = rand_i[j*self.grid_frame_number:(j+1)*self.grid_frame_number]
    
                # 使用随机索引从latents和control_image中选取对应的元素
                # prepare_key_grid_latents返回的是rand_indices中前四个索引对应frame组成的四合一大图
                latents_keyframe, _ = fu.prepare_key_grid_latents(latents, self.grid, self.grid, rand_indices)
                control_keyframe, _ = fu.prepare_key_grid_latents(control_image, self.grid, self.grid, rand_indices)
                # 将选取的元素添加到对应的列表中
                latents_l.append(latents_keyframe)
                controls_l.append(control_keyframe)
                # 将随机索引添加到randx列表中
                randx.extend(rand_indices)
            # 复制randx列表到rand_i变量中
            rand_i = randx.copy()
            # 将所有洗牌后的latents拼接成一个张量 latents里面是一个个四合一的大图
            latents = torch.cat(latents_l, dim=0)
            # 将所有洗牌后的control_image拼接成一个张量
            control_image = torch.cat(controls_l, dim=0)
            # 根据rand_i中的索引重新排序indices列表 rand_i里面是四个一组的索引值，这里把他变成一个一个的索引值
            indices = [indices[i] for i in rand_i]
            # 返回洗牌后的latents、indices和control_image
            return latents, indices, control_image
    
    @torch.no_grad()
    def batch_denoise(self, latents, control_image, indices, t, guidance_scale, current_sampling_percent):
            
            """
            对输入的latents和控制图像进行批量去噪处理。该函数首先将控制图像和latents按照批次大小进行分割，然后遍历每个批次的控制图像，为每个批次的latents创建一个文本嵌入向量，并对其进行去噪处理。最后，将所有处理后的latents和对应的控制图像拼接起来，返回处理后的latents、原始索引和controls。
            
            Parameters:
            latents (torch.Tensor): 待处理的latents。
            control_image (torch.Tensor): 用于控制去噪过程的控制图像。
            indices (list): 原始索引列表。
            t (int): 时间步。
            guidance_scale (float): 指导比例。
            current_sampling_percent (float): 当前采样百分比。
            
            Returns:
            tuple: 返回一个元组，包含处理后的latents、原始索引和controls。
            """        
            # 初始化两个空列表，用于存储处理后的latents和controls    
            latents_l, controls_l = [], []
            
            # 将控制图像按照批次大小进行分割
            control_split = control_image.split(self.batch_size, dim=0)
            
            # 将latents按照批次大小进行分割
            latents_split = latents.split(self.batch_size, dim=0)
            
            # 遍历每个批次的控制图像
            for idx in range(len(control_split)):
                # 创建一个文本嵌入向量，其中包含无条件和有条件的嵌入向量，长度与当前批次的latents相同
                txt_embed = torch.cat([self.uncond_embeddings] * len(latents_split[idx]) + [self.cond_embeddings] * len(latents_split[idx])) 
                
                # 对当前批次的latents和对应的控制图像进行去噪步骤，得到去噪后的latents
                latents = self.denoising_step(latents_split[idx], control_split[idx], txt_embed, t, guidance_scale, current_sampling_percent)
                
                # 将处理后的latents添加到latents_l列表中
                latents_l.append(latents)
                
                # 将对应的控制图像添加到controls_l列表中
                controls_l.append(control_split[idx])
            
            # 将latents_l列表中的所有元素按照维度0进行拼接，得到最终的latents
            latents = torch.cat(latents_l, dim=0)
            
            # 将controls_l列表中的所有元素按照维度0进行拼接，得到最终的controls
            controls = torch.cat(controls_l, dim=0)
            
            # 返回处理后的latents、原始索引和controls
            return latents, indices, controls
    
    @torch.no_grad()
    def reverse_diffusion(self, latents=None, control_image=None, guidance_scale=7.5, indices=None):
            """
                反向扩散过程。该函数接受潜在变量、控制图像、指导比例和索引作为参数，然后进行反向扩散过程。在每个时间步，根据当前采样百分比，对潜在变量、控制图像和索引进行相应的处理。最后返回处理后的潜在变量、索引和控制图像。
            
                Parameters:
                latents (torch.Tensor): 一个包含潜在变量的张量。默认为None。
                control_image (torch.Tensor): 一个包含控制图像的张量。默认为None。
                guidance_scale (float): 指导比例，用于控制反向扩散过程中的噪声级别。默认为7.5。
                indices (list): 一个包含索引的列表。默认为None。
            
                Returns:
                torch.Tensor: 返回处理后的潜在变量。
                list: 返回处理后的索引。
                torch.Tensor: 返回处理后的控制图像。
            """    
            # 设置调度器的时间步长
            self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            print("self.scheduler.timesteps=",self.scheduler.timesteps)
            # 使用CUDA进行自动类型转换
            with torch.autocast('cuda'):
                # 遍历调度器的所有时间步长
                for i, t in tqdm(enumerate(self.scheduler.timesteps), desc='reverse_diffusion'):
                    # 将索引转换为列表
                    indices = list(indices)
                    # 计算当前采样百分比
                    current_sampling_percent = i / len(self.scheduler.timesteps)
    
                    # 如果需要打乱潜在变量
                    if self.is_shuffle:
                        latents, indices, control_image = self.shuffle_latents(latents, control_image, indices,i)
                        
                    # 如果当前采样百分比大于等于条件步骤开始的百分比
                    if self.cond_step_start < current_sampling_percent:
                        # 对潜在变量、控制图像和索引进行批量去噪处理
                        latents, indices, controls = self.batch_denoise(latents, control_image, indices, t, guidance_scale, current_sampling_percent)
                    else:
                        # 对潜在变量、控制图像和索引进行批量去噪处理，但指导比例为0
                        latents, indices, controls = self.batch_denoise(latents, control_image, indices, t, 0.0, current_sampling_percent)
    
            # 返回处理后的潜在变量、索引和控制图像
            return latents, indices, controls

    @torch.no_grad()
    def encode_imgs(self, img_torch):
            """
            对输入的图像进行编码。该函数首先将输入的图像按照指定的批次大小进行分割，然后对每个分割后的图像进行编码，最后将所有编码结果拼接起来返回。
            
            Parameters:
            img_torch (Tensor): 一个待编码的图像张量。
            
            Returns:
            Tensor: 返回一个包含所有编码结果的张量。
            
            注意：这个函数依赖于VAE模型的encode方法和config属性中的scaling_factor属性。
            """    
            # 初始化一个空列表用于存储每个分割后的图像的编码结果
            latents_l = []
            
            # 将输入的图像张量按照指定的批次大小进行分割
            splits = img_torch.split(self.batch_size_vae, dim=0)
            
            # 遍历每个分割后的图像
            for split in splits:
                # 将分割后的图像数据范围从[0, 1]映射到[-1, 1]
                image = 2 * split - 1
                
                # 使用VAE模型的encode方法对图像进行编码，并获取其潜在分布
                posterior = self.vae.encode(image).latent_dist
                
                # 计算潜在分布的均值，并根据config属性中的scaling_factor属性进行缩放
                latents = posterior.mean * self.vae.config.scaling_factor
                
                # 将缩放后的潜在分布添加到列表中
                latents_l.append(latents)
            
            # 将所有编码结果拼接起来并返回
            return torch.cat(latents_l, dim=0)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
            """
            将潜在变量解码为图像。该函数接受一个torch.Tensor类型的参数，将其分割成多个批次，然后使用VAE模型的decode方法将每个批次的潜在变量解码为图像。最后将所有解码后的图像拼接在一起并返回。
            
            Parameters:
            latents (torch.Tensor): 一个包含潜在变量的torch.Tensor对象。
            
            Returns:
            torch.Tensor: 返回一个包含所有解码后图像的torch.Tensor对象。
            """    
            # 初始化一个空列表用于存储解码后的图像
            image_l = []
            # 将输入的潜在变量按照指定的批次大小进行分割
            splits = latents.split(self.batch_size_vae, dim=0)
            # 遍历每个分割后的潜在变量批次
            for split in splits:
                # 使用VAE模型的decode方法对当前批次的潜在变量进行解码，得到解码后的图像
                image = self.vae.decode(split / self.vae.config.scaling_factor, return_dict=False)[0]
                # 对解码后的图像进行归一化处理，使其值范围在[0, 1]之间
                image = (image / 2 + 0.5).clamp(0, 1)
                # 将处理后的图像添加到列表中
                image_l.append(image)
            # 将列表中的所有图像拼接在一起，形成一个完整的张量，并返回
            return torch.cat(image_l, dim=0)




    @torch.no_grad()
    def controlnet_pred(self, latent_model_input, t, text_embed_input, controlnet_cond):
            """
                该函数用于预测控制网络。它接收四个参数：latent_model_input（潜在模型输入），t（时间），text_embed_input（文本嵌入输入）和controlnet_cond（控制网络条件）。首先，它会调用控制网络来获取down_block_res_samples和mid_block_res_sample。然后，它会使用unet对latent_model_input、t、text_embed_input进行操作，并返回噪声预测。
            
                Parameters:
                latent_model_input (Tensor): 潜在模型的输入。
                t (int): 时间。
                text_embed_input (Tensor): 文本嵌入的输入。
                controlnet_cond (Tensor): 控制网络的条件。
            
                Returns:
                Tensor: 返回噪声预测。
            """    
            # 调用控制网络，获取down_block_res_samples和mid_block_res_sample
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embed_input,
                controlnet_cond=controlnet_cond,
                conditioning_scale=1,
                return_dict=False,
            )
            # 使用unet对latent_model_input、t、text_embed_input进行操作，并返回噪声预测
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embed_input,
                cross_attention_kwargs={},
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]
            return noise_pred

    @torch.no_grad()
    def ddim_inversion(self, latents, control_batch, indices):
        """
            该函数实现了DDIM反演过程。首先，它从指定的路径加载预训练的模型，然后使用DDIMScheduler设置时间步长。接着，
            根据时间步长进行反向迭代，计算每个时间步长的alpha_prod_t和alpha_prod_t_prev值，并更新latents。
            最后，将结果保存到指定路径。
        
            Parameters:
            latents (torch.Tensor): 待处理的张量。
            control_batch (torch.Tensor): 控制批次的张量。
            indices (list): 索引列表。
        
            Returns:
            torch.Tensor: 返回处理后的张量。
            list: 返回索引列表。
            torch.Tensor: 返回控制批次的张量。
        """    
        k = None
        els = os.listdir(self.inverse_path)  # 获取指定路径下的所有文件名
        els = [el for el in els if el.endswith('.pt')]  # 筛选出以'.pt'结尾的文件名
    
        for k,inv_path in enumerate(sorted(els, key=lambda x: int(x.split('.')[0]))):  # 对文件名进行排序并遍历
            latents[k] = torch.load(os.path.join(self.inverse_path, inv_path)).to(device=self.device)  # 加载预训练模型并移动到指定设备
    
        self.inverse_scheduler = DDIMScheduler.from_config(self.scheduler_config)  # 从配置中创建DDIMScheduler对象
        self.inverse_scheduler.set_timesteps(self.num_inversion_step, device=self.device)  # 设置时间步长
        self.timesteps = reversed(self.inverse_scheduler.timesteps)  # 获取倒序的时间步长
        print("self.timesteps=",self.timesteps)
        if k == (latents.shape[0]-1):  # 如果k等于latents的最后一个索引
            return latents, indices, control_batch  # 直接返回latents、indices和control_batch
        inv_cond = torch.cat([self.inv_uncond_embeddings] * 1 + [self.inv_cond_embeddings] * 1)[1].unsqueeze(0)  # 拼接无条件和有条件的嵌入向量
        for i, t in enumerate(tqdm(self.timesteps)):  # 遍历时间步长
            
            alpha_prod_t = self.inverse_scheduler.alphas_cumprod[t]  # 获取当前时间步长的alpha_prod_t值
            alpha_prod_t_prev = (self.inverse_scheduler.alphas_cumprod[self.timesteps[i - 1]] if i > 0 else self.inverse_scheduler.final_alpha_cumprod)  # 获取前一个时间步长的alpha_prod_t_prev值
            
            if k is not None:  # 如果k不为空
                if len(latents[:k+1].shape) == 3:  # 如果latents的形状为3维
                    latents[:k+1] = latents[:k+1].unsqueeze(0)  # 在第0维上增加一个维度
            latents_l = [] if k is None else [latents[:k+1]]  # 如果k为空，则latents_l为空列表，否则为包含latents[:k+1]的列表
            latents_split = latents.split(self.inv_batch_size, dim=0) if k is None else latents[k+1:].split(self.inv_batch_size, dim=0)  # 将latents分割成多个批次
            control_batch_split = control_batch.split(self.inv_batch_size, dim=0) if k is None else control_batch[k+1:].split(self.inv_batch_size, dim=0)  # 将control_batch分割成多个批次
            for idx in range(len(latents_split)):  # 遍历每个批次
                cond_batch = inv_cond.repeat(latents_split[idx].shape[0], 1, 1)  # 重复无条件和有条件的嵌入向量
                latents = self.ddim_step(latents_split[idx], t, cond_batch, alpha_prod_t, alpha_prod_t_prev, control_batch_split[idx])  # 执行ddim步骤
                latents_l.append(latents)  # 将更新后的latents添加到latents_l列表中
            latents = torch.cat(latents_l, dim=0)  # 将latents_l列表中的latents在0维上拼接起来
            
        for k,i in enumerate(latents):  # 遍历latents
            torch.save(i.detach().cpu(), f'{self.inverse_path}/{str(k).zfill(5)}.pt')  # 将处理后的latents保存到指定路径
    
        return latents, indices, control_batch  # 返回处理后的latents、indices和control_batch
    
   
    def ddim_step(self, latent_frames, t, cond_batch, alpha_prod_t, alpha_prod_t_prev, control_batch):
            
            """
                根据给定的参数计算并更新latent_frames的值。该函数首先计算当前时间步和前一个时间步的均值mu和mu_prev，以及标准差sigma和sigma_prev。然后根据是否提供控制反转来决定eps的计算方式。最后，计算预测的x0值，并更新latent_frames的值。
            
                Parameters:
                latent_frames (tensor): 当前的时间步的latent frames。
                t (int): 当前的时间步。
                cond_batch (tensor): 条件批次。
                alpha_prod_t (float): 当前时间步的alpha乘积。
                alpha_prod_t_prev (float): 前一个时间步的alpha乘积。
                control_batch (tensor): 控制批次。
            
                Returns:
                tensor: 返回更新后的latent_frames。
            """            
            # 计算当前时间步的均值mu，它是alpha_prod_t的平方根
            mu = alpha_prod_t ** 0.5
            # 计算前一个时间步的均值mu_prev，它是alpha_prod_t_prev的平方根
            mu_prev = alpha_prod_t_prev ** 0.5
            # 计算当前时间步的标准差sigma，它是(1 - alpha_prod_t)的平方根
            sigma = (1 - alpha_prod_t) ** 0.5
            # 计算前一个时间步的标准差sigma_prev，它是(1 - alpha_prod_t_prev)的平方根
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
            # 根据是否提供控制反转来决定eps的计算方式
            if self.give_control_inversion:
                # 如果提供了控制反转，使用controlnet_pred方法计算eps
                eps = self.controlnet_pred(latent_frames, t, text_embed_input=cond_batch, controlnet_cond=control_batch)
            else:
                # 如果没有提供控制反转，使用unet方法计算eps
                eps = self.unet(latent_frames, t, encoder_hidden_states=cond_batch, return_dict=False)[0]
            # 计算预测的x0值，它是latent_frames减去前一个时间步的标准差乘以eps后除以前一个时间步的均值
            pred_x0 = (latent_frames - sigma_prev * eps) / mu_prev
            # 更新latent_frames的值，它是当前时间步的均值乘以预测的x0加上当前时间步的标准差乘以eps
            latent_frames = mu * pred_x0 + sigma * eps
            # 返回更新后的latent_frames
            return latent_frames
    

    def process_image_batch(self, image_pil_list):
            """
            处理图像批次。该函数接受一个PIL图像列表，将其转换为torch张量，并保存到指定路径。如果指定路径下已有控制图像和原始图像的torch张量，则直接加载；否则，将PIL图像预处理为控制图像，然后转换为torch张量，并保存。
            
            Parameters:
            image_pil_list (list): 一个包含PIL图像的列表。
            
            Returns:
            img_torch (torch.Tensor): 转换后的原始图像torch张量。
            control_torch (torch.Tensor): 转换后的控制图像torch张量。
            """    
            # 如果控制图像路径下存在文件，则直接加载控制图像和原始图像的torch张量
            if len(os.listdir(self.controls_path)) > 0:
                control_torch = torch.load(os.path.join(self.controls_path, 'control.pt')).to(self.device)
                img_torch = torch.load(os.path.join(self.controls_path, 'img.pt')).to(self.device)
            else:
                # 初始化两个空列表用于存储控制图像和原始图像的torch张量
                image_torch_list = []
                control_torch_list = []
                # 遍历输入的PIL图像列表
                for image_pil in image_pil_list:
                    # 获取PIL图像的宽度和高度
                    width, height = image_pil.size
                    # 对PIL图像进行预处理，生成控制图像
                    control_pil = self.preprocess_control_grid(image_pil)
                    # 根据控制图像、宽度和高度准备控制图像
                    control_image = self.prepare_control_image(control_pil, width, height)
                    # 将控制图像添加到控制图像列表中
                    control_torch_list.append(control_image)
                    # 将PIL图像转换为torch张量，并添加到原始图像列表中
                    image_torch_list.append(ipu.pil_img_to_torch_tensor(image_pil))
                # 将控制图像列表中的torch张量按行拼接，并移动到指定设备上
                control_torch = torch.cat(control_torch_list, dim=0).to(self.device)
                # 将原始图像列表中的torch张量按行拼接，并移动到指定设备上
                img_torch = torch.cat(image_torch_list, dim=0).to(self.device)
                # 将控制图像和原始图像的torch张量保存到指定路径
                torch.save(control_torch, os.path.join(self.controls_path, 'control.pt'))
                torch.save(img_torch, os.path.join(self.controls_path, 'img.pt'))
                
            # 返回原始图像和控制图像的torch张量
            return img_torch, control_torch
        
    def order_grids(self, list_of_pils, indices):
            """
            将PIL图像列表转换为帧列表。该函数接受一个PIL图像列表和一个索引列表作为参数，将每个PIL图像转换为帧，并根据索引列表的顺序返回这些帧。
            
            Parameters:
            list_of_pils (List[PIL.Image.Image]): 一个包含多个PIL图像的列表。
            indices (List[int]): 一个整数列表，表示转换后的帧在最终结果中的顺序。
            
            Returns:
            List[Frame]: 返回一个帧列表，其中每个帧对应于输入列表中的一个PIL图像，顺序由索引列表决定。
            """    
            # 初始化一个空列表用于存储转换后的帧
            k = []
            # 遍历输入的PIL图像列表
            for i in range(len(list_of_pils)):
                # 调用fu.pil_grid_to_frames方法将每个PIL图像转换为帧，并将结果添加到k列表中
                k.extend(fu.pil_grid_to_frames(list_of_pils[i], self.grid))
                
            # 根据索引列表的顺序重新排列k列表中的帧，并返回新的帧列表
            frames = [k[indices.index(i)] for i in np.arange(len(indices))]    
            return frames


    @torch.autocast(dtype=torch.float16, device_type='cuda')  
    def batched_denoise_step(self, x, t, indices):
            """
            对输入的x进行批量去噪处理。该函数首先随机选择一批样本，然后对这些样本进行去噪处理，最后对所有样本进行去噪处理并返回结果。
            
            Parameters:
            x (torch.Tensor): 待处理的输入数据，其形状为(batch_size, *)。
            t (int): 时间步，用于确定去噪模型的状态。
            indices (list): 索引列表，用于确定每个样本在去噪过程中的位置。
            
            Returns:
            denoised_latents (torch.Tensor): 经过批量去噪处理后的输出数据，其形状为(batch_size, *)。
            """    
            # 获取配置中的批处理大小
            batch_size = self.config["batch_size"]
            # 初始化一个空列表，用于存储去噪后的样本
            denoised_latents = []
            # 生成一个随机索引数组，用于从输入数据中选择样本
            pivotal_idx = torch.randint(batch_size, (len(x)//batch_size,)) + torch.arange(0,len(x),batch_size) 
            
            # 对选定的样本进行去噪处理
            self.denoise_step(x[pivotal_idx], t, indices[pivotal_idx])
            # 遍历输入数据的批次，并对每个批次进行去噪处理
            for i, b in enumerate(range(0, len(x), batch_size)):
                # 将去噪后的样本添加到列表中
                denoised_latents.append(self.denoise_step(x[b:b + batch_size], t, indices[b:b + batch_size]))
            # 将所有去噪后的样本拼接成一个张量
            denoised_latents = torch.cat(denoised_latents)
            # 返回去噪后的张量
            return denoised_latents

    @torch.no_grad()
    def __preprocess_inversion_input(self, init_latents, control_batch):
            """
            对输入的初始潜在变量和控制批次进行预处理。该函数首先将每个初始潜在变量和控制批次展平，然后将其拼接在一起，最后按照总帧数进行切分。
            
            Parameters:
            init_latents (torch.Tensor): 一个包含多个初始潜在变量的张量。
            control_batch (torch.Tensor): 一个包含多个控制批次的张量。
            
            Returns:
            tuple: 返回一个元组，其中包含两个张量，分别是处理后的初始潜在变量和控制批次。
            """    
            # 使用列表推导式遍历初始潜在变量，对每个元素调用flatten_grid方法进行展平操作，并将结果存储在list_of_flattens中
            list_of_flattens = [fu.flatten_grid(el.unsqueeze(0), self.grid) for el in init_latents]
            # 将list_of_flattens中的所有张量沿着最后一个维度（dim=-1）拼接起来，得到init_latents
            init_latents = torch.cat(list_of_flattens, dim=-1)
            # 将init_latents沿着最后一个维度（dim=-1）按照self.total_frame_number进行切分，并沿着第一个维度（dim=0）拼接起来
            init_latents = torch.cat(torch.chunk(init_latents, self.total_frame_number, dim=-1), dim=0)
            
            # 对控制批次执行类似的操作
            control_batch_flattens = [fu.flatten_grid(el.unsqueeze(0), self.grid) for el in control_batch]
            control_batch = torch.cat(control_batch_flattens, dim=-1)
            control_batch = torch.cat(torch.chunk(control_batch, self.total_frame_number, dim=-1), dim=0)
            
            # 返回处理后的初始潜在变量和控制批次
            return init_latents, control_batch
    
    @torch.no_grad()
    def __postprocess_inversion_input(self, latents_inverted, control_batch):
            """
            对输入的倒置潜在变量和控制批次进行后处理。该函数首先将每个样本的倒置潜在变量和控制批次按照网格帧数进行切分，然后使用unflatten_grid函数将切分后的变量重新排列成原始的形状，最后将所有样本的结果合并成一个大的张量。
            
            Parameters:
            latents_inverted (torch.Tensor): 一个包含倒置潜在变量的张量。
            control_batch (torch.Tensor): 一个包含控制批次的张量。
            
            Returns:
            tuple: 返回一个元组，其中包含两个元素，第一个元素是处理后的倒置潜在变量，第二个元素是处理后的控制批次。
            """    
            latents_inverted = torch.cat([fu.unflatten_grid(torch.cat([a for a in latents_inverted[i*self.grid_frame_number:(i+1)*self.grid_frame_number]], dim=-1).unsqueeze(0), self.grid) for i in range(self.sample_size)] , dim=0)
            control_batch = torch.cat([fu.unflatten_grid(torch.cat([a for a in control_batch[i*self.grid_frame_number:(i+1)*self.grid_frame_number]], dim=-1).unsqueeze(0), self.grid) for i in range(self.sample_size)] , dim=0)
            return latents_inverted, control_batch
    
    
    @torch.no_grad()
    def __call__(self, input_dict):
            # 设置随机种子
            set_seed_lib(input_dict['seed'])
            
            # 从输入字典中获取参数
            self.grid_size = input_dict['grid_size']
            self.sample_size = input_dict['sample_size']
            
            # 计算网格帧数和总帧数
            self.grid_frame_number = self.grid_size * self.grid_size
            self.total_frame_number = (self.grid_frame_number) * self.sample_size
            self.grid = [self.grid_size, self.grid_size]
            
            # 从输入字典中获取其他参数
            self.cond_step_start = input_dict['cond_step_start']
            
            self.controlnet_guidance_start = input_dict['controlnet_guidance_start']
            self.controlnet_guidance_end = input_dict['controlnet_guidance_end']
            self.controlnet_conditioning_scale = input_dict['controlnet_conditioning_scale']
            
            self.positive_prompts = input_dict['positive_prompts']
            self.negative_prompts = input_dict['negative_prompts']
            self.inversion_prompt = input_dict['inversion_prompt']
            
            self.batch_size = input_dict['batch_size']
            self.inv_batch_size = self.batch_size * self.grid_size * self.grid_size
            self.batch_size_vae = input_dict['batch_size_vae']
    
            # 从输入字典中获取其他参数
            self.num_inference_steps = input_dict['num_inference_steps']
            self.num_inversion_step = input_dict['num_inversion_step']
            self.inverse_path = input_dict['inverse_path']
            self.controls_path = input_dict['control_path']
            
            # 从输入字典中获取其他参数
            self.is_ddim_inversion = input_dict['is_ddim_inversion']
            self.is_shuffle = input_dict['is_shuffle']
            self.give_control_inversion = input_dict['give_control_inversion']
            
            # 从输入字典中获取其他参数
            self.guidance_scale = input_dict['guidance_scale']
            
            # 生成索引列表
            indices = list(np.arange(self.total_frame_number))
            print("start_preprocessor")
            start_time=time.time()
            # 处理图像批次并编码图像
            img_batch, control_batch = self.process_image_batch(input_dict['image_pil_list'])
            init_latents_pre = self.encode_imgs(img_batch)
            # 深度图构建完成，释放 annotator 显存给后续 ControlNet 生成
            pu.unload_preprocess_models()
            
            # 创建调度器并设置时间步长
            self.scheduler = DDIMScheduler.from_config(self.scheduler_config)
            self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            # 获取文本嵌入
            self.inv_cond_embeddings, self.inv_uncond_embeddings = self.get_text_embeds(self.inversion_prompt, "")
            # 根据是否使用DDIM反转，进行相应的预处理、反转和后处理操作
            end_time=time.time()
            use_time=end_time-start_time
            print("use_time_preprocessor=",use_time)
            print("start_Inversion")
            start_time1=time.time()
            if self.is_ddim_inversion:
                
                init_latents, control_batch = self.__preprocess_inversion_input(init_latents_pre, control_batch)
                latents_inverted, indices, control_batch = self.ddim_inversion(init_latents, control_batch, indices)
                latents_inverted, control_batch = self.__postprocess_inversion_input(latents_inverted, control_batch)
            else:
                init_latents_pre = torch.cat([init_latents_pre], dim=0) 
                noise = torch.randn_like(init_latents_pre)
                latents_inverted = self.scheduler.add_noise(init_latents_pre, noise, self.scheduler.timesteps[:1])
            end_time1=time.time()
            use_time1=end_time1-start_time1
            print("use_time_Inversion=",use_time1)
            print("start_Inference")
            start_time2=time.time()
            # 获取文本嵌入
            self.cond_embeddings, self.uncond_embeddings = self.get_text_embeds(self.positive_prompts, self.negative_prompts)
            # 反向扩散过程
            latents_denoised, indices, controls = self.reverse_diffusion(latents_inverted, control_batch, self.guidance_scale, indices=indices)
            end_time2=time.time()
            use_time2=end_time2-start_time2
            print("use_time_Inference=",use_time2)
            # 解码潜在变量并生成图像
            image_torch = self.decode_latents(latents_denoised)
            # 对图像进行排序并返回结果
            ordered_img_frames = self.order_grids(ipu.torch_to_pil_img_batch(image_torch), indices)
            ordered_control_frames = self.order_grids(ipu.torch_to_pil_img_batch(controls), indices)
            return ordered_img_frames, ordered_control_frames
    

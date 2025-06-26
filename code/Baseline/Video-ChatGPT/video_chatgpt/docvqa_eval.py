from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
import torch
import json

#add new packages as below
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from video_chatgpt.eval.model_utils import initialize_model, load_video
import argparse
import numpy as np
import os

# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"



def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens


def video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len):
    """
    Run inference using the Video-ChatGPT model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Video-ChatGPT model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens.

    Returns:
    dict: Dictionary containing the model's output.
    """

    # Prepare question string for the model
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    inputs = tokenizer([prompt])

    # Preprocess video frames and get image tensor
    image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

    # Move image tensor to GPU and reduce precision to half
    image_tensor = image_tensor.half().cuda()

    # Generate video spatio-temporal features
    with torch.no_grad():
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
    video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # Define stopping criteria for generation
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    # Check if output is the same as input
    n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    # Decode output tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Clean output string
    outputs = outputs.strip().rstrip(stop_str).strip()

    return outputs


# 视频多个segment，不给video_text和speak_text
def load_data(args):
    os.environ['CUDA_VISIBLE_DEVICE']=args.gpu_id
    base_path = "/opt/ml/input/data/wrwwang/"
    # task_types = ["dev", "test", "S95"]
    task_types = [args.dataset_type]
    structured_data = []
    for task_type in task_types:
        data_path = base_path + "processed/{}_final.json".format(task_type)
        data = json.load(open(data_path, 'r'))
        for video_id, video_data in data.items():
            video_type = video_id.split('_')[0]
            instruction = "Go through the video. You are able to understand the visual and audio content provided by the user and respond to the question."
            for i in range(len(video_data)):
                segment = video_data[i]
                segment_id = "{}_{}".format(video_id, i)
                video_text = segment["video_text"]
                speak_text = segment["speak_text"]
                # v1 不给video_text和speak_text v2 给video_text不给speak_text v3 给speak_text不给video_text v4 都给
                
                QA_v1, QA_v2, QA_v3, QA_v4 = [], [], [], []
                for qa_pair in segment["QA_pairs"]:
                    video_instruction = "The text in the video is: " + video_text
                    speak_instruction = "The speaker in the video says: " + speak_text
                    question_v1 = " Answer the question: " + qa_pair["question"]
                    question_v2 = video_instruction + ". | Answer the question: " + qa_pair["question"]
                    question_v3 = speak_instruction + ". | Answer the question: " + qa_pair["question"]
                    question_v4 = video_instruction + ". | " + speak_instruction + ". | Answer the question: " + qa_pair["question"]
                    answer = qa_pair["answer"]
                    QA_v1.append({"question": question_v1, "answer": answer})
                    QA_v2.append({"question": question_v2, "answer": answer})
                    QA_v3.append({"question": question_v3, "answer": answer})
                    QA_v4.append({"question": question_v4, "answer": answer})
                    
                structured_data.append({
                    "task_type": task_type,
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "video_type": video_type,
                    "video": base_path + "segment_videos/{}/{}/{}/{}.mp4".format(task_type, video_type, video_id, segment_id),
                    "text_data": {
                        "instruction": instruction,
                        "v1": QA_v1,
                        "v2": QA_v2,
                        "v3": QA_v3,
                        "v4": QA_v4
                    }
                })

    predict_output = []

    # init model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path)
    conv_mode = args.conv_mode

    data_type = args.data_type
    for data in tqdm(structured_data):
        video_path = data["video"]
        img = None
        text_data = data["text_data"]
        if os.path.exists(video_path):
            video_frames = load_video(video_path)
        video_output = {}
        video_output[data_type] = []
        
        for qa in text_data[data_type]:
            question = qa["question"]
            target_answer = qa["answer"]
            predict_answer = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
            video_output[data_type].append({"question": question, "target_answer": target_answer, "predict_answer": predict_answer})

        predict_output.append({
            "task_type": data["task_type"],
            "video_id": data["video_id"],
            "segment_id": data["segment_id"],
            "video_type": data["video_type"],
            "predict_output": video_output
        })

        with open("/opt/ml/output/{}_video_chatgpt_output_{}.json".format(args.dataset_type ,data_type), "w") as f:
            json.dump(predict_output, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--projection_path", type=str, required=False, default="")
    parser.add_argument("--conv_mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--data_type", type=str, required=False, default='v1')
    parser.add_argument("--gpu_id", type=str, required=False, default="0")
    parser.add_argument("--dataset_type", type=str, required=False, default="dev")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    load_data(args)

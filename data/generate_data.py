import os
import json

meta_data = json.load(open('./train_meta_info.json', 'r'))
qa_data = json.load(open('./S95_QA.json', 'r'))
segments_data = json.load(open('./S95_processed.json', 'r'))

video_time = {}
for video_meta in meta_data["videos"]:
    for segment in video_meta["segments"]:
        uttid = segment["uttid"]
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        video_time[uttid] = (start_time, end_time)

final_data = {}
for video_id, qa_info in qa_data.items():
    final_data[video_id] = []
    for i in range(len(qa_info)):
        segment = segments_data[video_id][i]
        video_ids = segment["video_id"]
        start_time = min([float(video_time[video_seg][0]) for video_seg in video_ids])
        end_time = max([float(video_time[video_seg][1]) for video_seg in video_ids])
        final_data[video_id].append({
            "video_ids": video_ids,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "video_text": qa_data[video_id][i]["video_text"],
            "speak_text": qa_data[video_id][i]["speak_text"],
            "QA_pairs": qa_data[video_id][i]["QA_pairs"]
        })

json.dump(final_data, open('data', 'w'), indent=4, ensure_ascii=False)
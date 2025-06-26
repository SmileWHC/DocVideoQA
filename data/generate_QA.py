import json
import time
import re
from gpt_utils import get_completion
from tqdm import tqdm

def generate_QA_prompt(speak_text, video_text):
    prompt = """
>>> Task:
As an AI assistant, you are observing a presentation video. The speaker is talking with a PowerPoint slides.
Given the text extracted from PowerPoint slides (denoted as video_string) and the transcribed text of the speaker's narration (denoted as speak_string) from a segmented video presentation, create QA pairs to assess a model's comprehension of the video's content. The questions should fall into three categories to test the model's ability to:
    1.Extract answers exclusively from the PPT text (video_string).
    2.Derive answers solely from the speaker's narration (speak_string).
    3.Integrate information from both the PPT text and speaker's narration to find answers.

Your questions should aim for definitive answers, allowing one to confidently affirm the presence or absence of information based on the video content. Include questions of the following nature:
    1.Information Retrieval: Questions about specific details mentioned in the text, such as names, titles, or method names presented. Example: "What is the name of the process outlined on slide five?"
    2.Content Understanding: Questions that require understanding the content, possibly involving reasoning, calculations, or filling in blanks, based on the information provided. Example: "What contaminant was mentioned as polluting the oceans due to nuclear accidents?"

Ensure the questions:
    1.Cover a variety of topics and cognitive levels to thoroughly evaluate the model's comprehension abilities.
    2.Are impartial, avoiding leading questions that might bias the answerer towards a particular answer.
    3.Are designed for complexity, provide detailed answers, including examples or reasoning steps for more convincing and structured responses.

Remember not to inquire about uncertain details, and when questions are complex, provide thorough answers with detailed examples or reasoning steps to enhance persuasiveness and coherence.
Your output should be in the following JSON format, containing an array of QA pairs with each object holding a question and its corresponding answer

>>> Example:
Example 1:
Input:
video_string: CHILDRENS MIRACLE NETWORK MIRACLE HOSPITALS NETWORK DANCE MARATHON CHANGE KIDS HEALTH CHANGE THE FUTURE EMILY ASHLEY TWO PRONGED APPROACH TWO OVERARCHING AREAS OF STEWARDSHIP THAT WE FOCUS ON INDIVIDUAL I TH INCLUDES RESOURCES AND PLANS IMPLEMENTED TH SOREE THE SOREE V TA FOR PARTICIPANTS AND STAFF TO USE THEY THEMSELVES ARE ENCOURAGED THANK THE THOSE WHO HAVE DONATED TO THEIR EFFORTS ORGANIZATIONAL WHAT IS DONE ON BEHALF OF ALL OF PDM
speak_string: SO WE TAKE A TWO PRONGED APPROACH TOSTEWARDING IN TERMS OF UM THAT FROM AN INDIVIDUAL POINT OF VIEW AND THAT FROM AN ORGANIZATIONAL STANDPOINTSO IN TERMS OF INDIVIDUAL STEWARDSHIP THIS REFERS TO THE RESOURCES AND PLANS THAT ARE IMPLEMENTED FOR PARTICIPANTS AND STAFF MEMBERS TO USE WHERE THEY THEMSELVES ARE ENCOURAGED TO THANK THOSE WHO HAVE DONATED TO THEIR FUNDRAISING EFFORTSSO THIS INCLUDES TEMPLATES AND OUTLINES THAT WE PROVIDE TO EVERY UM INDIVIDUAL INVOLVED IN P D M TO SEND OUT ONCE THEY RECEIVE DONATIONS UM FROM DIFFERENT INDIVIDUALSAND THEN ORGANIZATIONAL STEWARDSHIP WHICH REFERS TO THE STEWARDING EFFORTS THAT ARE DONE ON BEHALF OF ALL OF PIT DANCE MARATHONSSO THIS INCLUDES EMAILS THAT ARE SENT OUT FROM P D M TO ALL DONORS UM TWO SPECIFIC CATEGORIES OF DONORS THAT SORT OF THING
Output:
[{{"question": "How many pronged approaches are mentioned in the video.", "answer": "Two pronged approach is mentioned in the video."}}, {{"question": "The video mentions which two overarching areas they're focusing on", "answer": "Individual and Organizational"}}, {{"question": "What is the focus of individual stewardship?", "answer": "The focus of individual stewardship is on resources and plans implemented for participants and staff to use."}}, {{"question": "What is done on behalf of all of PDM?", "answer": "Organizational stewardship is done on behalf of all of PDM."}}]

Example 2:
Input:
video_string: CHILDRENS MIRACLE NETWORK MIRACLE NETWORK HOSPITALS DANCE MARATHON KAPPLER EMMA IMPLEMENT INTING EFFECTIVE STEWARDSHIP EMILY ASHLEY EMMA KAPPLER SHEHER SHEHER DIRECTOR OPERATIONS PRESIDENT UNIVERSITY OF PITTSBURGH UNIVERSITY OF I PITTSBURGH DANCE MARATHON DANCE MARATHON
speak_string: ALRIGHT I'M GONNA GET RID OF THE MEETING CONTROLS REALLY QUICKLY TOOALRIGHT HI GUYS I'M EMMAI'M EMILYAND WE ARE FROM PIT DANCE MARATHON AND TODAY WE'RE GOING TO TALK A LITTLE BIT ABOUT IMPLEMENTING EFFECTIVE STEWARDSHIP FOR YOUR DANCE MARATHON
Output:
[{{"question": "Who is the speaker", "answer": "EMMA Kappler and Emily Ashley"}}, {{"question": "What is the topic of the video", "answer": "Implementing effective stewardship"}}, {{"question": "What organization are they from", "answer": "DANCE MARATHON}}]

>>> Input:
video_string: {}
speak_string: {}

>>> Output:
    """.format(video_text, speak_text)
    return prompt

def generate_QA(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    output_file = data_path.replace('_processed', '_QA')
    output = {}
    for video_id, segements in tqdm(data.items()):
        output[video_id] = []
        for segement in tqdm(segements):
            speak_text = segement['speak_string']
            video_text = segement['video_string']
            prompt = generate_QA_prompt(speak_text, video_text)
            gpt_input = {
                'role': 'user',
                'content': prompt
            }
            while True:
                try:
                    res = get_completion(gpt_input)
                    res = json.loads(res)
                    output[video_id].append({"video_id": video_id, "video_text": video_text, "speak_text": speak_text, "QA_pairs": res})
                    with open(output_file, 'w') as f:
                        json.dump(output, f, indent=4)
                    break
                except Exception as e:
                    print(e)
                    match = re.search(r"Please retry after (\d+) seconds", str(e))
                    if match:
                        wait_seconds = int(match.group(1))
                        print(f"Rate limit exceeded, waiting for {wait_seconds} seconds...")
                    else:
                        wait_seconds = 20
                    time.sleep(wait_seconds)

generate_QA('./processed/dev_processed.json')
generate_QA('./processed/test_processed.json')
generate_QA("./processed/S95_processed.json")
generate_QA('./processed/L95_processed.json')

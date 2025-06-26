import openai
openai.api_type = 
openai.api_base = 
openai.api_version = 
openai.api_key = 

def get_completion(prompt):
    response = openai.ChatCompletion.create(
    engine="gpt-4-32k",                                            #can replace with "gpt-4-32k" or "gpt-35-turbo"
    messages = [prompt],
    temperature=0.7,
    max_tokens=10000,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)
    return response['choices'][0]['message']['content']

prompt = {
    'role': 'user',
    'content': 'classify the sentiment of this sentence: "I am happy today". The sentiment is positive or negative? Answer: '
}
res = get_completion(prompt)
import requests
import json

def vllm_streaming_inference(prompt, temparature=0, max_tokens=128, top_k=-1, top_p=1, E_INST="[/INST]", stop_word="<|eot_id|>", url='0.0.0.0:9000'):
    pload = {"text_input": prompt, 
           "temperature": str(temparature),
           "max_tokens": str(max_tokens),
           "top_k": str(top_k),
           "top_p": str(top_p),
           "stream": True, 
           "stop": stop_word}
    
    
    response = requests.post('http://'+ url+'/v2/models/vllm_model/generate_stream', 
                             headers={"User-Agent": "vLLM Triton Client"}, 
                             json = pload, stream = True)
    
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\n\n"): 
        if chunk:
            output_str = chunk.decode("utf-8")
            data = json.loads(output_str.split("data: ")[1])
            yield data["text_output"]
            
def format_chat_prompt_llama_mistral(message, chat_history, ktexts, instruction):
    prompt = f"<s> [INST] \n{instruction}\n"
    for turn in chat_history:
        user_message, bot_message = turn
        if not user_message:
            continue
        else:
            prompt = f"{prompt}\nConversation history: {user_message} [/INST]{bot_message} </s>[INST]\n"
    if ktexts:
        prompt = f"{prompt}Context:" + ktexts + "\n"
    prompt=f"{prompt}Question: {message} Answer: [/INST]\n"
    return prompt        
            
ktexts = "Assign the Right Amount of Compute Power to Users, Automatically Run:AI’s Kubernetes-based software platform for orchestration of containerized AI workloads enables GPU clusters to be utilized for different Deep Learning workloads dynamically - from building AI models, to training, to inference. With Run:AI, jobs at any stage get access to the compute power they need, automatically. Run:AI’s compute management platform speeds up data science initiatives by pooling available resources and then dynamically allocating resources based on need - maximizing accessible compute."


chat_history = [["Run:ai is based on which orchestrator ?", "Kubernetes"]]

system = "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."

message = "Describe run:ai in detail ?"


prompt = format_chat_prompt_llama_mistral(message, chat_history, ktexts, system)


stream = vllm_streaming_inference(prompt, max_tokens=4096)
chat_history = chat_history + [[message, ""]]

acc_text = ""
for idx, text_token in enumerate(stream):
    acc_text += text_token
    last_turn = list(chat_history.pop(-1))
    last_turn[-1] += acc_text
    chat_history = chat_history + [last_turn]
    acc_text = ""
    

print(last_turn[-1])


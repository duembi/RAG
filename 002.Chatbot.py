import ollama


messages = [

    {"role" : "system", "content" : "your are an asistant. You asist about industry and technology data"},

]


while True :
    user_input = input("How can i help you ?")
        
    if user_input == "out" : 
        
        break
        
    messages.append({"role" : "user", "content" : user_input})
    
    llm_ans = ollama.chat(model ="gemma3:4b", keep_alive = "10m", messages = messages)
    print(llm_ans.message.content)
    messages.append({"role": "assistant", "content": llm_ans.message.content})

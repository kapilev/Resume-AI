import google.generativeai as genai
genai.configure(api_key="AIzaSyDjOzaPbtVwOrhfRa4ULujAKo_2vxK3H5g")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
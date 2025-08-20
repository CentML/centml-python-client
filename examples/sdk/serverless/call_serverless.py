from openai import OpenAI

client = OpenAI(
    api_key="Your serverless API key",
    base_url="https://api.centml.com/openai/v1",
)

# Define your question
user_question = "How does CentML improve your AIOps?"

completion = client.chat.completions.create(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_question}  # Add the user's question
    ],
    max_tokens=2000,
    temperature=0.7,
    top_p=1,
    n=1,
    stream=False,
    frequency_penalty=0,
    presence_penalty=0.5,
    stop=[]
)

print(completion.choices[0].message)
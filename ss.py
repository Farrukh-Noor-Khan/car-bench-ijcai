import os
from litellm import completion

# Do not paste your key directly in code.
# Set GEMINI_API_KEY in terminal instead.

try:
    response = completion(
        model="gemini/gemini-3.1-flash-lite",
        messages=[
            {
                "role": "user",
                "content": "Reply with the word 'Success' if you can read this."
            }
        ]
    )

    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("\nERROR TRIGGERED:")
    print(str(e))
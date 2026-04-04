#!/bin/bash
# Run CodeBoarding with Hugging Face Inference API (OpenAI-compatible)

# Use HF_TOKEN as OpenAI key
export OPENAI_API_KEY="$HF_TOKEN"
export OPENAI_BASE_URL="https://api-inference.huggingface.co/v1"

# Run CodeBoarding
uvx codeboarding "$@"

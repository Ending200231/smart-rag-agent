"""Quick Start example for Smart RAG Agent."""

from smart_rag import Agent

# Make sure you've built the index first:
#   smart-rag index --url https://fastapi.tiangolo.com --config config.yaml

# Initialize the agent
agent = Agent(index_dir="./index")

# Example 1: Simple retrieval question
print("=" * 60)
print("Example 1: Retrieval question")
print("=" * 60)
response = agent.ask("FastAPI 中如何定义路径参数？")
print(f"\nAnswer:\n{response.text}")
print(f"\nRoute: {response.trace[0].action}")
print(f"Sources: {len(response.sources)} docs")

# Example 2: Direct answer (no retrieval needed)
print("\n" + "=" * 60)
print("Example 2: Direct answer (no retrieval)")
print("=" * 60)
response = agent.ask("什么是 REST API？")
print(f"\nAnswer:\n{response.text}")
print(f"\nRoute: {response.trace[0].action}")

# Example 3: Complex question (decompose + multi-search)
print("\n" + "=" * 60)
print("Example 3: Complex question (decompose)")
print("=" * 60)
response = agent.ask("FastAPI 的 Depends 和 BackgroundTasks 有什么区别？分别在什么场景下使用？")
print(f"\nAnswer:\n{response.text}")
print(f"\nRoute: {response.trace[0].action}")
print(f"Sources: {len(response.sources)} docs")

# Print full trace
print("\nFull trace:")
for step in response.trace:
    print(f"  [{step.node}] {step.action}: {step.detail} ({step.duration_ms}ms)")

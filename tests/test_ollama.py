import sys
from crewai import LLM, Agent, Task, Crew

print(f"[STATUS] Testing runtime with Python version: {sys.version}")

try:
    # Initialize connection to local Llama instance
    local_llm = LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")
    
    # Setup test agent
    tester = Agent(
        role="System Validator",
        goal="Confirm local model response logic is active.",
        backstory="A script verifying local execution health.",
        llm=local_llm
    )
    
    # Setup simple task
    task = Task(
        description="Say the word 'ACTIVE' and nothing else.",
        expected_output="The exact word confirmation.",
        agent=tester
    )
    
    crew = Crew(agents=[tester], tasks=[task])
    print("[STATUS] Sending test ping to local Llama model...")
    response = crew.kickoff()
    print(f"\n[SUCCESS] Model Output received: {response}")

except Exception as e:
    print(f"\n[FAILURE] Local runtime handshake error: {e}")
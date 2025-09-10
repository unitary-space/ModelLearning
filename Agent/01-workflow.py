from llama_index.core.workflow import(
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

class MyWorkFlow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello World!")

async def main():
    w = MyWorkFlow(timeout=10, verbose=True)

    result = await w.run()
    print(result)

if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
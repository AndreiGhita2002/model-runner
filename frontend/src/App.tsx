import {useState} from "react";
import {Box, Button, ScrollArea} from "@radix-ui/themes";
// @ts-ignore
// import {ModuleDisplay} from "./ModuleDisplay.tsx";
// @ts-ignore
import {get_times, post_run_model, PrettyPrintJson} from "./util.tsx";

export default function App() {
    const [runResult, setRunResult] = useState<JSON>({});
    const [times, setTimes] = useState<JSON>({});

    return (
        <div>
            <Button variant="solid" onClick={() => post_run_model("simple-net", setRunResult)}>
                Run SimpleNet
            </Button>
            <Button variant="solid" onClick={() => post_run_model("conv-next", setRunResult)}>
                Run ConvNext
            </Button>

            <h2> Last model output: </h2>
            <ScrollArea type="always" scrollbars="vertical" style={{ height: 400 }}>
                <Box p="2" pr="8">
                    <PrettyPrintJson data={runResult} />
                </Box>
            </ScrollArea>

            <Button variant="solid" onClick={() => get_times(setTimes)}>
                Request Time Logs
            </Button>

            <ScrollArea type="always" scrollbars="vertical" style={{ height: 800 }}>
                <Box p="2" pr="8">
                    <PrettyPrintJson data={times} />
                </Box>
            </ScrollArea>

            {/*<ModuleDisplay module_data={tel.data} depth={0} />*/}



        </div>
    );
}

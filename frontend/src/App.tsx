import {useState} from "react";
import {Button} from "@radix-ui/themes";
// @ts-ignore
// import {ModuleDisplay} from "./ModuleDisplay.tsx";
// @ts-ignore
import {get_times, post_run_model, PrettyPrintJson} from "./util.tsx";

export default function App() {
    const [runOut, setRunOut] = useState<JSON>({});
    const [times, setTimes] = useState<JSON>({});

    return (
        <div>
            <Button variant="solid" onClick={() => post_run_model("simple-net", setRunOut)}>
                Run SimpleNet
            </Button>
            <Button variant="solid" onClick={() => post_run_model("conv-next", setRunOut)}>
                Run ConvNext
            </Button>

            <h2> Last model output: </h2>
            <PrettyPrintJson data={runOut} />

            <Button variant="solid" onClick={() => get_times(setTimes)}>
                Request Time Logs
            </Button>

            <PrettyPrintJson data={times} />

            {/*<ModuleDisplay module_data={tel.data} depth={0} />*/}



        </div>
    );
}


export type ModuleData = {
    time: {
        start: number,
        end: number,
        elapsed: number,
    },
    module_name: string,
    // module: any,
    children: [ModuleData],
}

function getModuleBorderStyle(depth: number) {
    const borderStr = "1mm solid ";
    const borderColour = [
        "blue",
        "green",
        "yellow",
        "red",
        "pink",
        "purple",
        "cyan"
    ]
    const colour = borderColour[depth % borderColour.length];
    return borderStr + colour;
}

interface ModuleDisplayArgs {
    module_data: ModuleData,
    depth: number,
}

export function ModuleDisplay({ module_data, depth }: ModuleDisplayArgs) {
    return (
        <div class='ModuleDisplay' style={{ border: getModuleBorderStyle(depth) }}>

            <span>Module name: {module_data.module_name}</span> <br/>
            <span><strong>elapsed time:</strong> {module_data.time.elapsed / 1000000} ms </span> <br/>
            {/*<span>start time: {module_data.time.start}</span> <br/>*/}
            {/*<span>end time: {module_data.time.end}</span> <br/>*/}

            {/* Children: */}
            {module_data.children.map((child)  => (
                <ModuleDisplay
                    module_data={child}
                    depth={depth + 1}
                />
            ))}
        </div>
    );
}

import './App.css';
import {useEffect, useState} from "react";
import axios from 'axios';
// @ts-ignore
import {ModuleDisplay} from "./ModuleDisplay.tsx";
// import Collapsible from "react-collapsible";
import { Collapsible } from "radix-ui";


function PrettyPrintJson(data: JSON | any) {
    return (
        <div>
            <pre>{
                JSON.stringify(data, null, 2)
            }</pre>
        </div>
    );
}

export default function App() {
    const [tel, setTel] = useState<JSON>({'status': 'Requesting...'});
    const [loading, setLoading] = useState<boolean>(true);
    const [open, setOpen] = useState<boolean>(false);

    const SIMPLE_NET_URL = 'http://127.0.0.1:5000/api/simple-net-test/'
    const CONV_NET_URL = 'http://127.0.0.1:5000/api/conv-net-test/'

    useEffect (() => {
        const fetchTel = async () => {
            return await axios.get(SIMPLE_NET_URL);
        }
        fetchTel().then(r => {
            setTel(r);
            setLoading(false);
        }).catch(console.error)
    }, []);

    return (
        <div>
            {loading && <div>Loading...</div>}
            {!loading && <div>
                <ModuleDisplay module_data={tel.data} depth={0} />
                {/* TODO: try the other collapsible component */}
                {/*<Collapsible.Root*/}
                {/*    open={open}*/}
                {/*    onOpenChange={setOpen}*/}
                {/*>*/}
                {/*    <Collapsible.Content>*/}
                        <PrettyPrintJson data={tel.data} />
                {/*    </Collapsible.Content>*/}
                {/*</Collapsible.Root>*/}
            </div>}
        </div>
    );
}

import axios from 'axios';
import {SetStateAction} from "react";


export function PrettyPrintJson(data: JSON | any) {
    return (
        <div>
            <pre>{
                JSON.stringify(data, null, 2)
            }</pre>
        </div>
    );
}

export function post_run_model(model_name: string, responseSetter: SetStateAction<JSON>) {
    const POST_RUN_MODEL_URL = "http://127.0.0.1:5000/api/run-model/";
    let url = POST_RUN_MODEL_URL + model_name;

    axios.post(url, {}).then(r => responseSetter(r.data));
}

export function get_times(responseSetter: SetStateAction<JSON>) {
    const GET_TIMES_URL = "http://127.0.0.1:5000/api/times";

    axios.get(GET_TIMES_URL).then(r => responseSetter(r.data));
}
import React from 'react';
import ReactDOM from 'react-dom/client';
import './style/index.css';
import "@radix-ui/themes/styles.css";
import {Theme} from "@radix-ui/themes";
// @ts-ignore
import App from './App.tsx';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <Theme>
            <App />
        </Theme>
    </React.StrictMode>
);

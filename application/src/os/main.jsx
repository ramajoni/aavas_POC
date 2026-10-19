import React from "react";
import { createRoot } from "react-dom/client";
import { OsApp } from "./OsApp.jsx";
import "../styles.css";
import "./os.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <OsApp />
  </React.StrictMode>,
);

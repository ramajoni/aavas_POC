import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.jsx";
import { ExplainRoute, parseExplainHash } from "./LoanExplain.jsx";
import "./styles.css";

// A loan explanation opens in its own window with no workspace chrome around it.
const explain = parseExplainHash(location.hash);

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    {explain ? <ExplainRoute initial={explain} /> : <App />}
  </React.StrictMode>,
);

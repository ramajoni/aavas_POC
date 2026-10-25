import { createContext, useContext } from 'react';

// Which pages the signed-in role may open. Pages read it to leave out links to
// pages the role cannot open, instead of offering a link that only answers
// "not available for your role". Outside a provider every page is allowed.
export const AccessContext = createContext(() => true);
export const useCanOpen = () => useContext(AccessContext);

import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import UploadDog from "./pages/UploadDog";
import UploadCat from "./pages/UploadCat";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="upload-dog" element={<UploadDog />} />
        <Route path="upload-cat" element={<UploadCat />} />
      </Route>
    </Routes>
  );
}

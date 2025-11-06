import { useNavigate } from "react-router-dom";
import "./Home.css";

export default function Home() {
  const navigate = useNavigate();

  const handleSelect = (species: string) => {
    if (species === "dog") navigate("/upload-dog");
    else if (species === "cat") navigate("/upload-cat");
  };

  return (
    <div className="home">
      <div className="intro">
        <h1 className="title">당신의 반려동물, 지금 어떤 기분일까요?</h1>
        <p className="subtitle">
          PawTune이 표정을 읽고 그 마음에 어울리는 음악을 선물해요.
        </p>
      </div>

      <div className="select-section">
        <h2>당신의 반려동물은?</h2>
        <div className="pet-options">
          <div className="pet-card dog" onClick={() => handleSelect("dog")}>
            <img src="/dog.jpeg" alt="강아지" className="pet-image" />
          </div>
          <div className="pet-card cat" onClick={() => handleSelect("cat")}>
            <img src="/cat.jpeg" alt="고양이" className="pet-image" />
          </div>
        </div>
      </div>
    </div>
  );
}

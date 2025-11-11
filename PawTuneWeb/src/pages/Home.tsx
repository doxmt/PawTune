import { useNavigate } from "react-router-dom";
import "./Home.css";

export default function Home() {
  const navigate = useNavigate();

  const handleStart = () => {
    navigate("/upload-dog");
  };

  return (
    <div className="home">
      <div className="intro">
        <h1 className="title">당신의 반려동물, 지금 어떤 기분일까요?</h1>
        <p className="subtitle">
          PawTune이 강아지의 표정을 분석하고, 감정에 어울리는 음악을 추천해요.
        </p>
      </div>

      {/* 강아지 카드 섹션 */}
      <div className="pet-preview">
        <div className="pet-card dog" onClick={handleStart}>
          <img src="/dog.jpeg" alt="강아지" className="pet-image" />
        </div>
      </div>

      {/* 시작 버튼 */}
      <button className="start-btn" onClick={handleStart}>
        지금 시작하기 🐾
      </button>
    </div>
  );
}

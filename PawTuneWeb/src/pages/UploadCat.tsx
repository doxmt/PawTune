import { useState } from "react";
import "./Upload.css";

export default function UploadCat() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [result, setResult] = useState<string>("");

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      setResult(""); // 새 이미지 선택 시 결과 초기화
    }
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    // TODO: 여기에 Flask 서버 AI 호출 (고양이용)
    // 예시로 임시 결과 표시
    setResult("😺 고양이가 편안해 보입니다!");
  };

  return (
    <div className="upload cat-upload">
      <h1>🐱 고양이 표정 분석</h1>
      <p>AI가 고양이의 표정을 분석해 감정을 예측합니다.</p>

      {/* 이미지 & 결과 섹션 */}
      <div className="analysis-section">
        {selectedImage ? (
          <img src={selectedImage} alt="preview" className="preview-image" />
        ) : (
          <div className="placeholder">이미지를 업로드해주세요 🐾</div>
        )}
        {result && <p className="result-text">{result}</p>}
      </div>

      {/* 버튼 영역 */}
      <div className="btn-group">
        <label htmlFor="catFile" className="btn">
          {selectedImage ? "다시 선택" : "사진 선택"}
        </label>
        <input
          id="catFile"
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          style={{ display: "none" }}
        />

        <button
          className="btn analyze"
          disabled={!selectedImage}
          onClick={handleAnalyze}
        >
          분석 시작
        </button>
      </div>
    </div>
  );
}

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_naive_bayes_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Prediksi Gaya Belajar", page_icon="🎓")
st.title("🎓 Prediksi Gaya Belajar Mahasiswa")
st.markdown("Masukkan data mahasiswa untuk memprediksi gaya belajar menggunakan model Naive Bayes.")

gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
age = st.number_input("Umur", min_value=15, max_value=60, value=20)
study_hours = st.slider("Jam Belajar per Minggu", 0, 60, 20)
online_courses = st.slider("Jumlah Kursus Online yang Selesai", 0, 50, 5)
participation = st.selectbox("Berpartisipasi dalam Diskusi?", options=label_encoders["Participation_in_Discussions"].classes_)
assignment_completion = st.slider("Tingkat Penyelesaian Tugas (%)", 0, 100, 80)
exam_score = st.slider("Nilai Ujian (%)", 0, 100, 70)
attendance = st.slider("Kehadiran (%)", 0, 100, 90)
use_tech = st.selectbox("Menggunakan Teknologi Edukasi?", options=label_encoders["Use_of_Educational_Tech"].classes_)
stress_level = st.selectbox("Tingkat Stres", options=label_encoders["Self_Reported_Stress_Level"].classes_)
social_media = st.slider("Jam Sosial Media per Minggu", 0, 70, 10)
sleep = st.slider("Jam Tidur per Malam", 0, 12, 7)
final_grade = st.selectbox("Nilai Akhir", options=label_encoders["Final_Grade"].classes_)

input_data = {
    "Gender": label_encoders["Gender"].transform([gender])[0],
    "Age": age,
    "Study_Hours_per_Week": study_hours,
    "Online_Courses_Completed": online_courses,
    "Participation_in_Discussions": label_encoders["Participation_in_Discussions"].transform([participation])[0],
    "Assignment_Completion_Rate (%)": assignment_completion,
    "Exam_Score (%)": exam_score,
    "Attendance_Rate (%)": attendance,
    "Use_of_Educational_Tech": label_encoders["Use_of_Educational_Tech"].transform([use_tech])[0],
    "Self_Reported_Stress_Level": label_encoders["Self_Reported_Stress_Level"].transform([stress_level])[0],
    "Time_Spent_on_Social_Media (hours/week)": social_media,
    "Sleep_Hours_per_Night": sleep,
    "Final_Grade": label_encoders["Final_Grade"].transform([final_grade])[0],
}

input_df = pd.DataFrame([input_data])

if st.button("🔍 Prediksi Gaya Belajar"):
    prediction = model.predict(input_df)[0]
    style = label_encoders['Preferred_Learning_Style'].inverse_transform([prediction])[0]
    st.success(f"🎯 Gaya Belajar yang Diprediksi: **{style}**")

    learning_styles_info = {
        "Visual": {
            "deskripsi": "Tipe belajar Visual menyukai penggunaan gambar, warna, dan diagram. Mereka mudah memahami informasi yang disajikan secara visual.",
            "strategi": [
                "Gunakan peta konsep atau mind map",
                "Tandai poin penting dengan warna",
                "Tonton video pembelajaran"
            ],
            "media": "Infografik, video tutorial, flashcard visual"
        },
        "Auditory": {
            "deskripsi": "Tipe belajar Auditory belajar lebih baik melalui mendengarkan. Mereka menyukai diskusi, podcast, atau penjelasan lisan.",
            "strategi": [
                "Belajar sambil mendengarkan rekaman suara",
                "Ikut diskusi kelompok",
                "Gunakan teknik membaca keras"
            ],
            "media": "Podcast, audio book, penjelasan verbal"
        },
        "Kinesthetic": {
            "deskripsi": "Tipe belajar Kinesthetic suka belajar melalui praktik langsung dan aktivitas fisik.",
            "strategi": [
                "Lakukan praktik atau eksperimen",
                "Gunakan model nyata",
                "Belajar sambil bergerak (walk and learn)"
            ],
            "media": "Simulasi, eksperimen langsung, role play"
        },
        "Reading/Writing": {
            "deskripsi": "Tipe belajar ini menyukai teks dan catatan. Mereka senang membaca buku dan membuat ringkasan.",
            "strategi": [
                "Buat catatan dengan tulisan sendiri",
                "Baca buku atau artikel",
                "Gunakan daftar dan definisi"
            ],
            "media": "Buku, jurnal, catatan pribadi"
        }
    }

    info = learning_styles_info.get(style, None)

    if info:
        st.markdown("## 🧠 Tentang Gaya Belajar Ini")
        st.info(info["deskripsi"])

        st.markdown("### 📌 Strategi Belajar yang Cocok")
        for strategi in info["strategi"]:
            st.write(f"- {strategi}")

        st.markdown("### 📚 Media yang Disarankan")
        st.success(info["media"])
    else:
        st.warning("Tidak ada informasi tambahan untuk gaya belajar ini.")

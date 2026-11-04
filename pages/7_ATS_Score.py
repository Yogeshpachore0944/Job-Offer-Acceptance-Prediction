# OfferSense AI
# pages/7_ATS_Score.py

import re
from io import BytesIO

import numpy as np
import streamlit as st
from utils.page_state import activate_page


activate_page("7_ATS_Score")

st.set_page_config(page_title="Resume ATS Score", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #0b0f17; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1200px; padding: 2rem 3rem 3rem; }
    .title { color: #f8fafc; font-size: 40px; font-weight: 750; }
    .subtitle { color: #94a3b8; font-size: 17px; margin-bottom: 24px; }
    .score-card { background: #111827; border: 1px solid #233047; border-radius: 18px; padding: 28px; margin: 20px 0; }
    .score-number { color: #f8fafc; font-size: 56px; font-weight: 800; line-height: 1; }
    .score-label { color: #f8fafc; font-size: 26px; font-weight: 750; margin-top: 10px; }
    .score-copy { color: #cbd5e1; font-size: 15px; line-height: 1.6; }
    .metric-card { background: #111827; border: 1px solid #1f2937; border-radius: 14px; padding: 18px; min-height: 100px; }
    .metric-label { color: #94a3b8; font-size: 13px; }
    .metric-value { color: #f8fafc; font-size: 25px; font-weight: 750; margin-top: 8px; }
    .recommendation { background: #0f172a; border-left: 4px solid #3b82f6; border-radius: 10px; color: #cbd5e1; padding: 14px 18px; margin: 8px 0; }
    .footer { color: #64748b; text-align: center; margin-top: 35px; }
    </style>
    """,
    unsafe_allow_html=True,
)


SECTION_ALIASES = {
    "Contact": ["contact", "email", "phone", "mobile", "linkedin"],
    "Summary": ["summary", "profile", "objective"],
    "Skills": ["skills", "technical skills", "core skills", "competencies"],
    "Experience": ["experience", "work experience", "employment", "work history"],
    "Education": ["education", "academic background", "qualifications"],
    "Projects": ["projects", "academic projects", "personal projects"],
    "Certifications": ["certifications", "certificates", "licenses"],
}

ROLE_PROFILES = {
    "Data Analyst": {
        "keywords": {"sql", "excel", "power bi", "tableau", "dashboard", "reporting", "kpi", "data cleaning", "eda"},
        "focus": "SQL, Excel, dashboards, data cleaning, reporting and business insights",
    },
    "Data Science": {
        "keywords": {"python", "pandas", "numpy", "scikit-learn", "machine learning", "predictive", "statistics", "feature engineering", "model evaluation"},
        "focus": "Python, statistics, feature engineering, predictive modelling and model evaluation",
    },
    "AI/ML": {
        "keywords": {"python", "scikit-learn", "tensorflow", "pytorch", "fastapi", "api", "deployment", "pipeline", "machine learning", "deep learning"},
        "focus": "model development, APIs, deployment, pipelines and production-oriented machine learning",
    },
}


def contains_phrase(text, phrase):
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def clean_text(text):
    return re.sub(r"\s+", " ", text.lower().replace("–", "-").replace("—", "-")).strip()


def tokens(text):
    return re.findall(r"[a-zA-Z][a-zA-Z+#.&'/-]{1,}", clean_text(text))


def extract_pdf_text(uploaded_file):
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PDF support requires PyMuPDF. Install it with: pip install pymupdf") from error

    uploaded_file.seek(0)
    document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in document)
    document.close()
    return text


def extract_docx_text(uploaded_file):
    from docx import Document

    uploaded_file.seek(0)
    document = Document(BytesIO(uploaded_file.read()))
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            value = " ".join(cell.text.strip() for cell in row.cells)
            if value:
                parts.append(value)
    return "\n".join(parts)


def extract_resume_text(uploaded_file):
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)
    if filename.endswith(".docx"):
        return extract_docx_text(uploaded_file)
    raise ValueError("Unsupported file format. Please upload a PDF or DOCX resume.")


def calculate_ats_score(resume_text):
    text = clean_text(resume_text)
    word_count = len(tokens(text))
    sections = {
        section: any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text) for alias in aliases)
        for section, aliases in SECTION_ALIASES.items()
    }
    contact = {
        "email": bool(re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", text)),
        "phone": bool(re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", text)),
        "linkedin": "linkedin" in text,
        "github": "github" in text,
    }

    if word_count >= 250:
        length_score = 40
    elif word_count >= 150:
        length_score = 30
    elif word_count >= 80:
        length_score = 20
    else:
        length_score = 10

    action_words = {
        "developed", "built", "created", "designed", "implemented", "analyzed",
        "managed", "improved", "optimized", "led", "automated", "delivered",
        "achieved", "reduced", "increased", "deployed", "tested", "maintained",
    }
    action_terms = sorted(
        word for word in action_words
        if re.search(r"(?<!\w)" + word + r"(?!\w)", text)
    )
    action_count = len(action_terms)
    quantified_results = len(re.findall(r"(?:\d+%|\$\s?\d[\d,.]*|\b\d+(?:\.\d+)?\s?(?:years?|months?|users?|clients?|projects?|hours?|days?)\b)", text))
    role_matches = {
        role: sorted(keyword for keyword in profile["keywords"] if contains_phrase(text, keyword))
        for role, profile in ROLE_PROFILES.items()
    }
    role_scores = {
        role: round(len(matches) / len(ROLE_PROFILES[role]["keywords"]) * 100, 1)
        for role, matches in role_matches.items()
    }
    project_count = len(re.findall(r"\b(?:project|dashboard|system|application|pipeline|model)\b", text))
    content_score = min(100, length_score + min(30, action_count * 5) + sum(sections.values()) / len(sections) * 20)
    structure_score = sum(sections.values()) / len(sections) * 100
    contact_score = sum(contact.values()) / len(contact) * 100
    length_component = min(100, word_count / 5)
    score = float(np.clip(
        structure_score * 0.35
        + contact_score * 0.20
        + content_score * 0.25
        + length_component * 0.20,
        0,
        100,
    ))

    return {
        "score": score,
        "sections": sections,
        "contact": contact,
        "content_score": content_score,
        "structure_score": structure_score,
        "contact_score": contact_score,
        "length_component": length_component,
        "word_count": word_count,
        "action_count": action_count,
        "action_terms": action_terms,
        "quantified_results": quantified_results,
        "role_matches": role_matches,
        "role_scores": role_scores,
        "project_count": project_count,
    }


def verdict_for_score(score):
    if score >= 80:
        return "Excellent", "This resume is well structured and likely to parse cleanly in most ATS systems."
    if score >= 60:
        return "Good", "This resume is usable for ATS screening, but a few improvements could make it stronger."
    return "Not Good", "This resume has important ATS-readiness gaps and should be improved before applying."


def category_ranking(result):
    missing_sections = [name for name, found in result["sections"].items() if not found]
    missing_contact = [name for name, found in result["contact"].items() if not found]
    categories = [
        {
            "name": "Contact and links",
            "score": result["contact_score"],
            "detail": (
                f"Missing: {', '.join(missing_contact)}."
                if missing_contact else "The resume gives recruiters the email, phone and professional links they need."
            ),
        },
        {
            "name": "Section structure",
            "score": result["structure_score"],
            "detail": (
                f"The resume needs clearer sections for: {', '.join(missing_sections)}."
                if missing_sections else "The resume follows a clear structure with the standard sections recruiters expect."
            ),
        },
        {
            "name": "Content and impact",
            "score": result["content_score"],
            "detail": (
                f"Uses {', '.join(result['action_terms'])}; "
                f"the resume gives {result['quantified_results']} measurable result(s) to support the work."
                if result["action_terms"] else
                "The resume bullets need stronger achievement language to show the candidate's contribution."
            ),
        },
        {
            "name": "Length and coverage",
            "score": result["length_component"],
            "detail": f"The resume uses {result['word_count']} words to present the candidate's experience and skills.",
        },
        {
            "name": "Action language",
            "score": min(100, result["action_count"] / 8 * 100),
            "detail": (
                "The experience and project bullets need a clear pattern: action + task + measurable result."
                if result["quantified_results"] == 0 else
                f"The resume includes {result['quantified_results']} measurable result(s); connect them to the strongest experience and project bullets."
            ),
        },
    ]
    return sorted(categories, key=lambda category: category["score"])


def candidate_focus(result):
    weakest = category_ranking(result)[0]["name"]
    focus = {
        "Contact and links": "Make the resume reachable first: complete the missing contact or profile links.",
        "Section structure": "Make the resume easier to scan first: add the missing standard sections with clear headings.",
        "Content and impact": "Make the experience credible first: turn responsibilities into outcomes with numbers, scale or time saved.",
        "Length and coverage": "Add or remove content first so the resume gives enough evidence without repeating itself.",
        "Action language": "Rewrite the experience and project bullets first using action, task and result.",
    }
    return weakest, focus[weakest]


def recommendations(result):
    missing_sections = [name for name, found in result["sections"].items() if not found]
    missing_contact = [name for name, found in result["contact"].items() if not found]
    suggestions = []

    for category in category_ranking(result):
        if category["name"] == "Contact and links" and missing_contact:
            suggestions.append(
                f"Contact and links: add the missing {', '.join(missing_contact)} so recruiters can identify and reach you."
            )
        elif category["name"] == "Section structure" and missing_sections:
            suggestions.append(
                f"Section structure: add clear headings for {', '.join(missing_sections)} using standard ATS-friendly names."
            )
        elif category["name"] == "Content and impact" and result["content_score"] < 70:
            suggestions.append(
                "Content and impact: rewrite the three most important bullets to show what you did, how you did it and what changed because of it."
            )
        elif category["name"] == "Length and coverage" and result["word_count"] < 150:
            suggestions.append(
                f"Length and coverage: expand this {result['word_count']}-word resume with relevant responsibilities, projects and achievements."
            )
        elif category["name"] == "Length and coverage" and result["word_count"] > 700:
            suggestions.append(
                f"Length and coverage: reduce repetition in this {result['word_count']}-word resume and keep the strongest evidence."
            )
        elif category["name"] == "Action language" and result["action_count"] < 5:
            suggestions.append(
                "Action language: start each experience or project bullet with a specific verb, then add the task and a measurable result."
            )

    if result["quantified_results"] == 0:
        suggestions.append(
            "Evidence: add numbers wherever the resume allows it, such as percentage improvement, users supported, projects completed, time saved or revenue affected."
        )

    if not suggestions:
        suggestions.append("Keep the formatting simple and tailor the skills and achievements to each target role.")
    return suggestions


def resume_audit(result):
    strengths = []
    loopholes = []

    if result["contact"]["email"]:
        strengths.append("The resume includes a professional email address.")
    else:
        loopholes.append("Add a professional email address so recruiters can contact you.")

    if result["contact"]["phone"]:
        strengths.append("A phone number is present.")
    else:
        loopholes.append("Add a phone number so recruiters have a direct way to reach you.")

    if result["contact"]["linkedin"] or result["contact"]["github"]:
        strengths.append("The resume gives recruiters access to a professional profile.")
    else:
        loopholes.append("Add a LinkedIn or GitHub URL to support the claims made in the resume.")

    present_sections = [name for name, present in result["sections"].items() if present]
    missing_sections = [name for name, present in result["sections"].items() if not present]
    if len(present_sections) >= 5:
        strengths.append("The resume has a clear section structure that is easy for recruiters to scan.")
    if missing_sections:
        loopholes.append(f"Strengthen the resume structure by adding: {', '.join(missing_sections)}.")

    if result["action_count"] >= 5:
        strengths.append(f"The resume uses achievement-focused verbs such as {', '.join(result['action_terms'])}.")
    else:
        loopholes.append("Rewrite passive bullets with specific verbs that show the candidate's contribution.")

    if 150 <= result["word_count"] <= 700:
        strengths.append("The resume has enough content to communicate the candidate's background without unnecessary length.")
    elif result["word_count"] < 150:
        loopholes.append("The resume is short; add relevant projects, responsibilities and outcomes to show more evidence of ability.")
    else:
        loopholes.append("The resume is long; remove repetition and keep the strongest evidence of impact.")

    return strengths, loopholes


def job_readiness(result):
    required = result["contact"]["email"] and result["contact"]["phone"]
    important_sections = all(result["sections"].get(section) for section in ["Skills", "Experience", "Education"])
    if result["score"] >= 80 and required and important_sections:
        return "Ready to Apply", "The resume is generally ready for applications. Tailor its skills and achievements to each specific role."
    if result["score"] >= 60:
        return "Needs Improvements", "The resume can be used, but fix the listed loopholes before applying to improve recruiter and ATS readability."
    return "Not Ready Yet", "Important resume gaps should be fixed before using this resume for job applications."


def detailed_review(result):
    role_order = sorted(result["role_scores"], key=result["role_scores"].get, reverse=True)
    best_role = role_order[0]
    best_keywords = result["role_matches"][best_role]
    missing_sections = [name for name, present in result["sections"].items() if not present]
    missing_contact = [name for name, present in result["contact"].items() if not present]

    if result["quantified_results"] == 0:
        biggest_problem = "The resume describes responsibilities and tools, but it gives very little measurable proof of impact."
    elif result["project_count"] < 3:
        biggest_problem = "The resume needs more project evidence to prove that the listed skills can be applied in different situations."
    elif result["role_scores"][best_role] < 45:
        biggest_problem = "The resume is not yet positioned strongly enough for one target role because its job-specific keyword coverage is limited."
    else:
        biggest_problem = f"The strongest opportunity is to position the resume clearly for {best_role} instead of presenting several directions equally."

    summary = {
        "best_role": best_role,
        "best_keywords": best_keywords,
        "role_order": role_order,
        "biggest_problem": biggest_problem,
        "missing_sections": missing_sections,
        "missing_contact": missing_contact,
        "summary_assessment": (
            "The resume has a target-role summary that can be strengthened with specific tools, project evidence and the preferred job title."
            if result["sections"]["Summary"] else
            "The resume needs a targeted professional summary explaining who the candidate is, what they can do and which role they want."
        ),
        "project_assessment": (
            f"The resume refers to approximately {result['project_count']} project or build-related items. The strongest projects should show the problem, method, tools and outcome."
            if result["project_count"] else
            "The resume does not give enough project evidence. Add two or three projects with the problem, method, tools and outcome."
        ),
        "skills_assessment": (
            f"For {best_role}, the resume currently supports these keywords: {', '.join(best_keywords) if best_keywords else 'no strong role-specific keyword group yet'}."
        ),
        "quantification_assessment": (
            f"The resume includes {result['quantified_results']} measurable detail(s). Add verified metrics such as dataset size, users, accuracy, recall, time saved or dashboard KPIs."
        ),
    }
    return summary


def role_outlook(result, role):
    score = result["role_scores"][role]
    if score >= 70:
        level = "Strongest current fit"
    elif score >= 45:
        level = "Possible fit with targeted improvements"
    else:
        level = "Needs stronger evidence before targeting"
    keywords = result["role_matches"][role]
    return level, ", ".join(keywords) if keywords else "No strong matching keywords yet"


with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("---")
    st.title("Navigation")
    st.write("ATS Resume Score Module")
    st.markdown("---")  

st.markdown("<div class='title'>Resume ATS Score</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload your resume to see whether it is Excellent, Good, or Not Good for ATS screening.</div>", unsafe_allow_html=True)
st.info("Only the resume is analyzed. No Job Description is required.")

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"],
    key="ats_resume_file",
    help="Upload a text-based PDF or DOCX resume.",
)
analyze = st.button("Analyze Resume", use_container_width=True, type="primary")

if analyze:
    if resume_file is None:
        st.warning("Please upload a PDF or DOCX resume.")
    else:
        try:
            with st.spinner("Reading and analyzing your resume..."):
                resume_text = extract_resume_text(resume_file)
                if len(resume_text.strip()) < 80:
                    st.error("Very little text was extracted. Please use a text-based PDF or DOCX resume.")
                else:
                    st.session_state["resume_text"] = resume_text
                    st.session_state["ats_result"] = calculate_ats_score(resume_text)
        except Exception as error:
            st.error("Resume analysis failed.")
            st.exception(error)

if "ats_result" in st.session_state:
    result = st.session_state["ats_result"]
    verdict, explanation = verdict_for_score(result["score"])

    st.markdown("---")
    st.markdown(
        f"<div class='score-card'><div class='score-number'>{result['score']:.1f}<small> / 100</small></div>"
        f"<div class='score-label'>{verdict}</div><p class='score-copy'>{explanation}</p></div>",
        unsafe_allow_html=True,
    )

    review = detailed_review(result)
    st.markdown(
        f"This is a practical ATS-readiness assessment for **{review['best_role']}** and related entry-level roles. "
        "There is no universal ATS score: each employer and job description can value different keywords and evidence."
    )

    st.subheader("My scoring")
    scoring_rows = [
        ("ATS-friendly structure", result["structure_score"] / 10, 10, "Section order and standard headings"),
        ("Contact information", result["contact_score"] / 10, 10, "Reachability and professional links"),
        ("Professional summary", 10 if result["sections"]["Summary"] else 4, 10, review["summary_assessment"]),
        ("Technical skills", min(10, result["role_scores"][review["best_role"]] / 10), 10, "Keywords relevant to the strongest role fit"),
        ("Projects and evidence", min(20, result["project_count"] / 3 * 20), 20, review["project_assessment"]),
        ("Quantified achievements", min(10, result["quantified_results"] / 3 * 10), 10, review["quantification_assessment"]),
        ("Job targeting", min(5, result["role_scores"][review["best_role"]] / 20), 5, f"Strongest current direction: {review['best_role']}"),
        ("Resume length", min(10, result["length_component"] / 10), 10, f"{result['word_count']} words"),
    ]
    st.markdown(
        "| Area | Score | Assessment |\n|---|---:|---|\n" +
        "\n".join(f"| {area} | {score:.1f}/{maximum:g} | {assessment} |" for area, score, maximum, assessment in scoring_rows)
    )

    st.subheader("Biggest problem")
    st.markdown(f"> {review['biggest_problem']}\n\nThis should be the first improvement because it affects how quickly a recruiter can understand the candidate's value.")

    st.subheader("Professional summary")
    st.write(review["summary_assessment"])
    st.write(
        f"For this resume, the summary should connect the candidate's background to **{review['best_role']}** and mention relevant strengths such as **{', '.join(review['best_keywords']) if review['best_keywords'] else 'the strongest verified tools and projects'}**."
    )

    st.subheader("Projects and practical evidence")
    st.write(review["project_assessment"])
    st.write("Each project should answer four questions: what problem was addressed, which method was used, what tools were involved, and what result was achieved.")

    st.subheader("Skills and keyword positioning")
    st.write(review["skills_assessment"])
    st.write("Use only skills the candidate can explain in an interview. A targeted skills section is stronger than listing tools without project evidence.")

    st.subheader("Quantified achievements")
    st.write(review["quantification_assessment"])
    st.write("Do not invent numbers. Calculate the real values from the project or work and connect each number to a specific bullet.")

    st.subheader("Role outlook")
    st.markdown("| Target role | Outlook | Resume evidence |\n|---|---|---|\n" + "\n".join(
        f"| {role} | {role_outlook(result, role)[0]} | {role_outlook(result, role)[1]} |"
        for role in review["role_order"]
    ))

    st.subheader("Priority plan")
    priority_items = [
        f"Choose one primary direction, preferably {review['best_role']}, and tailor the title and summary to it.",
        "Rewrite the three most important project or experience bullets as action + method + verified result.",
        "Add a second or third substantial project if the resume currently relies on one main project.",
        "Move the strongest role-specific tools into a grouped skills section and support them with project evidence.",
        "Add professional profile URLs and remove low-value content that takes space away from projects or outcomes." if review["missing_contact"] else "Review LinkedIn and GitHub so they support the projects and skills claimed on the resume.",
    ]
    for index, item in enumerate(priority_items, start=1):
        st.markdown(f"**Priority {index}:** {item}")

    st.subheader("Relevant Resume Checks")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-card'><div class='metric-label'>Structure</div><div class='metric-value'>{result['structure_score']:.1f}%</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-label'>Contact Details</div><div class='metric-value'>{result['contact_score']:.1f}%</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-label'>Resume Length</div><div class='metric-value'>{result['word_count']} words</div></div>", unsafe_allow_html=True)

    st.subheader("Category ranking")
    st.caption("Categories are ranked from the weakest area to the strongest area for this resume.")
    for rank, category in enumerate(category_ranking(result), start=1):
        st.markdown(f"**{rank}. {category['name']} — {category['score']:.1f}%**")
        st.progress(int(round(category["score"])))
        st.caption(category["detail"])

    focus_category, focus_message = candidate_focus(result)
    st.subheader("What to focus on first")
    st.markdown(
        f"<div class='recommendation'><b>{focus_category}</b><br>{focus_message}</div>",
        unsafe_allow_html=True,
    )

    readiness_label, readiness_explanation = job_readiness(result)
    st.subheader("General Job Readiness")
    st.markdown(
        f"<div class='recommendation'><b>{readiness_label}</b><br>{readiness_explanation}</div>",
        unsafe_allow_html=True,
    )

    _, loopholes = resume_audit(result)
    st.subheader("Resume Loopholes")
    if loopholes:
        for loophole in loopholes:
            st.warning(loophole)
    else:
        st.markdown("- No major ATS-readiness gaps need attention right now.")

    st.subheader("Detailed Resume Information")
    present = [name for name, included in result["sections"].items() if included]
    missing = [name for name, included in result["sections"].items() if not included]
    st.write(f"Resume strengths: {', '.join(present) if present else 'The standard sections still need to be added'}.")
    st.write(f"Contact readiness: email {'is available' if result['contact']['email'] else 'needs to be added'} · phone {'is available' if result['contact']['phone'] else 'needs to be added'}.")
    st.write(f"Professional links: {sum([result['contact']['linkedin'], result['contact']['github']])}/2 are included · measurable results supporting the claims: {result['quantified_results']}.")
    if missing:
        st.write(f"Sections to strengthen: {', '.join(missing)}.")

    st.subheader("What To Improve")
    for suggestion in recommendations(result):
        st.markdown(f"<div class='recommendation'>{suggestion}</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='footer'>
        <h4> OfferSense AI </h4>
        <p>
        Machine Learning Based Job Offer Acceptance Prediction System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

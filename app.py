from flask import Flask, render_template, request, redirect, session, jsonify, send_file
import pandas as pd
import joblib
import os, json, io, re
from functools import wraps

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from utils.emails_otp import send_otp, verify_otp, otp_store
from utils.database import save_user, get_user_data, create_tables, save_prediction_data, get_prediction_by_id, update_user_profile, verify_user_password, set_user_password

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

try:
    import google.generativeai as genai
except ImportError:
    genai = None

app = Flask(__name__)
app.template_folder = "template"
app.secret_key = os.getenv("SECRET_KEY", "insuresence-secure-key-2026")

# Session Security Configuration
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

create_tables()

ASSISTANT_NAME = "InsureBot"

# Load Trained ML Models
premium_model = joblib.load("models/premium_model.joblib")
policy_model = joblib.load("models/policy_model.joblib")
policy_label_encoder = joblib.load("models/policy_label_encoder.joblib")
claim_model = joblib.load("models/claim_model.joblib")

# Gemini AI Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
use_ai = False
GEMINI_MODEL_NAME = "gemini-1.5-flash"

if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        use_ai = True
        print("[OK] Gemini AI Enabled")
    except Exception as e:
        print(f"[WARN] Could not initialize Gemini: {e}")

def _gen_gemini():
    if not use_ai:
        return None
    try:
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception:
        try:
            return genai.GenerativeModel("gemini-pro")
        except Exception:
            return None

def gemini_support(message, personal, context_data=None):
    if not use_ai:
        return "InsureBot support is currently operating in offline mode. Please contact human support."

    model = _gen_gemini()
    if not model:
        return "InsureBot service is temporarily unavailable. Please try again in a moment."

    name = personal.get("name", "User")
    
    context_str = ""
    if context_data:
        ud = context_data.get("user_data", {})
        context_str = f"""
User Profile Context:
- Age: {ud.get('age', 'N/A')}
- BMI: {ud.get('bmi', 'N/A')}
- Smoker: {'Yes' if ud.get('smoker') == 1 else 'No'}
- Latest Estimated Premium: ₹{context_data.get('premium', 'N/A')}
- Claim Approval Risk: {context_data.get('probability', 'N/A')}%
"""

    prompt = f"""
You are InsureBot, an expert AI customer support and health insurance advisor.
Respond clearly, politely, and helpfully to the user.

App Capabilities:
- Predict annual and monthly insurance premiums based on age, BMI, smoker status, dependents, region.
- Calculate claim approval probability and recommend tailored policy plans.
- Provide prediction history, interactive what-if savings simulator, and downloadable PDF reports.

User Name: {name}
{context_str}
User Question: "{message}"

Provide a direct, concise, and helpful response referencing their health metrics if relevant. Keep response brief and engaging.
"""
    try:
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        return "Support service unavailable at the moment. Please try again later."

def gemini_advice(context):
    fallback_response = {
        "health_assessment": ["Maintain a balanced diet and regular weekly physical activity.", "Schedule annual comprehensive health checkups."],
        "insurance_guidance": ["Review deductibles and coverage limits suitable for your lifestyle.", "Consider add-on critical illness riders for enhanced protection."],
        "recommended_policies": context.get("policies", [])[:3],
        "closing": "InsureBot is dedicated to helping you make smart, healthy, and financially sound choices."
    }

    if not use_ai:
        return fallback_response

    model = _gen_gemini()
    if not model:
        return fallback_response

    pd_data = context["personal_details"]
    ud_data = context["user_data"]

    name = pd_data.get("name", "User")
    bmi = ud_data.get("bmi", 22.0)
    smoker = "Yes" if ud_data.get("smoker") == 1 else "No"
    claim_prob = context.get("probability", 0.0)
    monthly = context.get("monthly_premium", 0.0)
    policies = sorted(context.get("policies", []), key=lambda x: x[1], reverse=True)

    prompt = f"""
You are {ASSISTANT_NAME}, an expert insurance risk analyst and health advisor.
Analyze the user's data and provide actionable recommendations in the exact structure below:

**Health Assessment:**
- Bullet point 1
- Bullet point 2

**Insurance Guidance:**
- Bullet point 1
- Bullet point 2

**Recommended Policies:**
- Policy Name (Probability%)

**Closing:**
A friendly, reassuring closing sentence.

User Profile:
- Name: {name}
- BMI: {bmi}
- Smoker: {smoker}
- Claim Probability: {claim_prob}%
- Monthly Estimated Premium: ₹{monthly}
- Recommended Policies: {', '.join([f'{p} ({pr}%)' for p, pr in policies[:3]])}
"""
    try:
        result = model.generate_content(prompt)
        text = result.text.strip()

        sections = {
            "health_assessment": [],
            "insurance_guidance": [],
            "recommended_policies": [],
            "closing": ""
        }

        current_section = None
        for line in text.split('\n'):
            line_str = line.strip()
            if not line_str:
                continue

            if "Health Assessment" in line_str:
                current_section = "health_assessment"
            elif "Insurance Guidance" in line_str:
                current_section = "insurance_guidance"
            elif "Recommended Policies" in line_str:
                current_section = "recommended_policies"
            elif "Closing" in line_str:
                current_section = "closing"
            elif current_section and line_str.startswith("-"):
                clean_item = line_str.lstrip("-* ").strip()
                if current_section == "recommended_policies":
                    match = re.search(r"^(.*?)\s*\((\d+(?:\.\d+)?)\%\)$", clean_item)
                    if match:
                        sections[current_section].append((match.group(1), float(match.group(2))))
                    else:
                        sections[current_section].append((clean_item, 100.0))
                else:
                    sections[current_section].append(clean_item)
            elif current_section == "closing":
                sections["closing"] += line_str + " "

        sections["closing"] = sections["closing"].strip()

        if not sections["health_assessment"]:
            sections["health_assessment"] = fallback_response["health_assessment"]
        if not sections["insurance_guidance"]:
            sections["insurance_guidance"] = fallback_response["insurance_guidance"]
        if not sections["recommended_policies"]:
            sections["recommended_policies"] = policies[:3]

        return sections
    except Exception as e:
        print(f"[WARN] Gemini Advice Error: {e}")
        return fallback_response

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "email" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

@app.route("/")
def root():
    return redirect("/home")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/dashboard")
@login_required
def dashboard():
    user_info = get_user_data(session["email"])
    predictions = user_info.get("predictions", [])
    total_predictions = len(predictions)
    
    last_prediction = predictions[-1] if predictions else None
    last_user_data = last_prediction.get("user_data", {}) if last_prediction else None

    return render_template(
        "dashboard.html",
        user_info=user_info,
        total_predictions=total_predictions,
        last_prediction=last_prediction,
        last_user_data=last_user_data
    )

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not email:
            return render_template("signup.html", error="Please enter a valid email address.")
        
        session["pending_email"] = email
        if password:
            session["pending_password"] = password

        send_otp(email)
        return redirect("/verify")
    return render_template("signup.html")

@app.route("/verify", methods=["GET", "POST"])
def verify():
    email = session.get("pending_email") or session.get("email")
    if not email:
        return redirect("/signup")

    otp_dev = None
    entry = otp_store.get(email)
    if isinstance(entry, dict):
        otp_dev = entry.get("code")

    if request.method == "POST":
        user_otp = request.form.get("otp", "").strip()
        success, message = verify_otp(email, user_otp)

        if success:
            pwd = session.pop("pending_password", None)
            save_user(email, password=pwd)
            session["email"] = email
            session.pop("pending_email", None)
            return redirect("/dashboard")
        
        return render_template("verify.html", email=email, otp=otp_dev, error=message)

    return render_template("verify.html", email=email, otp=otp_dev)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        login_type = request.form.get("login_type", "password")

        if not email:
            return render_template("login.html", error="Please enter your email.")

        if login_type == "password":
            if not password:
                return render_template("login.html", error="Please enter your password.")
            if verify_user_password(email, password):
                session["email"] = email
                return redirect("/dashboard")
            else:
                return render_template("login.html", error="Invalid email or password. You can also log in via OTP.")

        # Default to OTP login option
        session["pending_email"] = email
        send_otp(email)
        return redirect("/verify")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/home")

@app.route("/index")
@login_required
def index():
    user_info = get_user_data(session["email"])
    return render_template("index.html", user_info=user_info)

@app.route("/api/calculate_bmi", methods=["POST"])
def calculate_bmi_api():
    try:
        data = request.get_json()
        height_cm = float(data.get("height_cm", 0))
        weight_kg = float(data.get("weight_kg", 0))
        if height_cm > 0 and weight_kg > 0:
            bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
            category = "Underweight" if bmi < 18.5 else "Normal weight" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            return jsonify({"success": True, "bmi": bmi, "category": category})
    except Exception:
        pass
    return jsonify({"success": False, "error": "Invalid measurements"})

@app.route("/api/what_if_simulator", methods=["POST"])
def what_if_simulator():
    try:
        data = request.get_json() or {}
        age = int(data.get("age", 30))
        bmi = float(data.get("bmi", 24.5))
        smoker = int(data.get("smoker", 0))
        children = int(data.get("children", 0))
        sex = 1
        region = 1

        df = pd.DataFrame([{
            "age": age,
            "sex_encoded": sex,
            "bmi": bmi,
            "children": children,
            "smoker_encoded": smoker,
            "region_encoded": region
        }])

        sim_premium = float(premium_model.predict(df)[0])
        sim_premium = max(sim_premium, 1000.0)

        return jsonify({
            "success": True,
            "simulated_premium": round(sim_premium, 2)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    d = request.form
    region_map = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
    region = region_map.get(d.get("region", "").lower(), 0)
    smoker = 1 if d.get("smoker", "").lower() == "yes" else 0
    sex = 1 if d.get("sex", "").lower() == "male" else 0

    age = int(d.get("age", 25))
    bmi = float(d.get("bmi", 22.0))
    children = int(d.get("children", 0))

    df = pd.DataFrame([{
        "age": age,
        "sex_encoded": sex,
        "bmi": bmi,
        "children": children,
        "smoker_encoded": smoker,
        "region_encoded": region
    }])

    session["personal_details"] = {
        "name": d.get("name", "User"),
        "address": d.get("address", "N/A"),
        "blood_group": d.get("blood_group", "N/A")
    }

    update_user_profile(session["email"], name=d.get("name"), address=d.get("address"), blood_group=d.get("blood_group"))

    premium = float(premium_model.predict(df)[0])
    premium = max(premium, 1000.0)

    claim_df = df[["age", "bmi", "smoker_encoded", "region_encoded", "children"]]
    claim_prob = float(claim_model.predict_proba(claim_df)[0][1] * 100)
    claim = claim_prob > 50
    monthly = round(premium / 12, 2)

    policy_df = df[["age", "bmi", "smoker_encoded", "children"]]
    probs = policy_model.predict_proba(policy_df)[0]
    names = list(policy_label_encoder.classes_)
    policies = [(names[i], round(float(p) * 100, 2)) for i, p in enumerate(probs)]

    data = {
        "premium": round(premium, 2),
        "monthly_premium": monthly,
        "claim": claim,
        "probability": round(claim_prob, 2),
        "policies": policies,
        "personal_details": session["personal_details"],
        "user_data": {
            "age": age,
            "bmi": bmi,
            "smoker": smoker,
            "region": region,
            "children": children,
            "sex": sex
        }
    }

    data["advice"] = gemini_advice(data)
    save_prediction_data(session["email"], data)
    session["last_prediction"] = data

    return render_template("result.html", **data)

@app.route("/history")
@login_required
def history():
    user_info = get_user_data(session["email"])
    return render_template("history.html", predictions=user_info.get("predictions", []))

@app.route("/history_view/<int:pred_id>")
@login_required
def history_view(pred_id):
    data = get_prediction_by_id(pred_id, session["email"])
    if not data:
        return redirect("/history")
    return generate_pdf_response(data)

@app.route("/download_pdf")
@login_required
def download_pdf():
    data = session.get("last_prediction")
    if not data:
        user_info = get_user_data(session["email"])
        preds = user_info.get("predictions", [])
        if preds:
            data = preds[-1]
        else:
            return redirect("/history")
    return generate_pdf_response(data)

def generate_pdf_response(data):
    filename = f"{data['personal_details']['name'].replace(' ', '_')}_Insurance_Report.pdf"
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=1,
        spaceAfter=15
    )
    
    sub_title_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#64748b'),
        alignment=1,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2563eb'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=3,
        spaceAfter=3
    )

    story = [
        Paragraph("InsureSence Insurance Assessment Report", title_style),
        Paragraph(f"Generated for {data['personal_details'].get('name', 'User')} | InsureBot AI Advisor", sub_title_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=15),
    ]

    table_data = [
        [Paragraph("<b>Full Name:</b>", body_style), Paragraph(str(data['personal_details'].get('name', '')), body_style),
         Paragraph("<b>Annual Premium:</b>", body_style), Paragraph(f"Rs. {data.get('premium', 0.0):,.2f}", body_style)],
        [Paragraph("<b>Blood Group:</b>", body_style), Paragraph(str(data['personal_details'].get('blood_group', '')), body_style),
         Paragraph("<b>Monthly Payment:</b>", body_style), Paragraph(f"Rs. {data.get('monthly_premium', 0.0):,.2f}", body_style)],
        [Paragraph("<b>Address:</b>", body_style), Paragraph(str(data['personal_details'].get('address', '')), body_style),
         Paragraph("<b>Claim Risk:</b>", body_style), Paragraph(f"{data.get('probability', 0.0)}%", body_style)]
    ]

    t = Table(table_data, colWidths=[110, 160, 110, 160])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Recommended Policy Packages", heading_style))
    policies = data.get("policies", [])
    if policies:
        policy_table_data = [[Paragraph("<b>Policy Tier</b>", body_style), Paragraph("<b>Match Confidence</b>", body_style)]]
        for name, prob in policies:
            policy_table_data.append([Paragraph(name, body_style), Paragraph(f"{prob}%", body_style)])
        
        pt = Table(policy_table_data, colWidths=[300, 240])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(pt)

    story.append(Spacer(1, 15))

    story.append(Paragraph("AI Health & Financial Advisor Insights", heading_style))
    advice = data.get("advice", {})

    if advice.get("health_assessment"):
        story.append(Paragraph("<b>Health Assessment:</b>", body_style))
        for pt in advice["health_assessment"]:
            story.append(Paragraph(f"• {pt}", body_style))
        story.append(Spacer(1, 6))

    if advice.get("insurance_guidance"):
        story.append(Paragraph("<b>Insurance Guidance:</b>", body_style))
        for pt in advice["insurance_guidance"]:
            story.append(Paragraph(f"• {pt}", body_style))
        story.append(Spacer(1, 6))

    if advice.get("closing"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<i>{advice['closing']}</i>", body_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceAfter=10))
    story.append(Paragraph("<font size=8 color='#94a3b8'>Notice: This report is generated by InsureSence AI prediction engine based on statistical insurance models and health risk metrics. It serves as an informative estimation and should be validated with licensed insurance underwriters.</font>", body_style))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, download_name=filename, as_attachment=True, mimetype='application/pdf')

@app.route("/ask", methods=["GET", "POST"])
@login_required
def ask():
    answer = None
    user_info = get_user_data(session["email"])
    last_pred = user_info["predictions"][-1] if user_info.get("predictions") else None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer = gemini_support(question, session.get("personal_details", {}), context_data=last_pred)
    return render_template("ask.html", answer=answer, assistant_name=ASSISTANT_NAME)

@app.route("/support")
def support():
    return render_template("support.html", assistant_name=ASSISTANT_NAME)

@app.route("/support_api", methods=["POST"])
def support_api():
    req_data = request.get_json() or {}
    message = req_data.get("message", "").strip()
    
    last_pred = None
    if "email" in session:
        user_info = get_user_data(session["email"])
        if user_info.get("predictions"):
            last_pred = user_info["predictions"][-1]

    reply = gemini_support(message, session.get("personal_details", {}), context_data=last_pred)
    return jsonify({"reply": reply, "assistant_name": ASSISTANT_NAME})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting InsureSence Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

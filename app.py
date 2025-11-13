from flask import Flask, render_template, request, redirect, session, jsonify, send_file
import pandas as pd
import joblib
from utils.emails_otp import send_otp, verify_otp, otp_store
from utils.database import save_user, get_user_data, create_tables, save_prediction_data
from functools import wraps
from dotenv import load_dotenv
import os, json, io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

try:
    import google.generativeai as genai
except:
    genai = None


app = Flask(__name__)
app.template_folder = "template"
app.secret_key = "your-secret-key"
create_tables()

ASSISTANT_NAME = "InsureBot"

premium_model = joblib.load("models/premium_model.joblib")
policy_model = joblib.load("models/policy_model.joblib")
policy_label_encoder = joblib.load("models/policy_label_encoder.joblib")
claim_model = joblib.load("models/claim_model.joblib")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL_NAME = "gemini-2.5-flash"
use_ai = False

if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        use_ai = True
        print("✅ Gemini AI Enabled")
    except:
        print("❌ Could not initialize Gemini.")


def _gen_gemini():
    if not use_ai:
        return None
    try:
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except:
        return None


def gemini_support(message, personal):
    if not use_ai:
        return "InsureBot support currently offline."

    model = _gen_gemini()
    if not model:
        return "InsureBot is currently unavailable. Try later."

    name = personal.get("name", "User")

    prompt = f"""
You are InsureBot Support.
You respond clearly, politely, and professionally.

User Name: {name}
User Message: "{message}"

If the user is new:
- Tell them: Sign Up → Verify OTP → Login → Go to Predict Page

If the user needs help:
- Explain step-by-step with short sentences

Avoid long paragraphs.
"""

    try:
        res = model.generate_content(prompt)
        return res.text.strip()
    except:
        return "Support service unavailable at the moment."


def gemini_advice(context):
    if not use_ai:
        return {
            "health_assessment": ["AI advice temporarily unavailable."],
            "insurance_guidance": [],
            "recommended_policies": [],
            "closing": ""
        }

    model = _gen_gemini()
    if not model:
        return {
            "health_assessment": ["InsureBot is currently unavailable. Try later."],
            "insurance_guidance": [],
            "recommended_policies": [],
            "closing": ""
        }

    pd = context["personal_details"]
    ud = context["user_data"]

    name = pd["name"]
    bmi = ud["bmi"]
    smoker = "Yes" if ud["smoker"] else "No"
    region_map = {0:"Southwest",1:"Southeast",2:"Northwest",3:"Northeast"}
    region_label = region_map.get(ud["region"], "N/A")

    premium = context["premium"]
    monthly = context["monthly_premium"]
    claim_prob = context["probability"]
    policies = sorted(context["policies"], key=lambda x:x[1], reverse=True)

    prompt = f"""
You are {ASSISTANT_NAME}, a professional insurance advisor.

Provide advice in the following structured format:

**Health Assessment:**
- Bullet point 1
- Bullet point 2
- etc.

**Insurance Guidance:**
- Bullet point 1
- Bullet point 2
- etc.

**Recommended Policies:**
- Policy1 (prob%)
- Policy2 (prob%)
- etc.

**Closing:**
A polite closing message.

Context:
- Name: {name}
- BMI: {bmi}
- Smoker: {smoker}
- Claim Probability: {claim_prob}%
- Monthly Cost: ₹{monthly}
- Top Policies: {', '.join([f"{p} ({pr}%)" for p,pr in policies[:3]])}
"""

    try:
        result = model.generate_content(prompt)
        text = result.text.strip()

        # Parse the response into sections
        sections = {"health_assessment": [], "insurance_guidance": [], "recommended_policies": [], "closing": ""}

        current_section = None
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("**Health Assessment:**"):
                current_section = "health_assessment"
            elif line.startswith("**Insurance Guidance:**"):
                current_section = "insurance_guidance"
            elif line.startswith("**Recommended Policies:**"):
                current_section = "recommended_policies"
            elif line.startswith("**Closing:**"):
                current_section = "closing"
            elif current_section and line.startswith("- "):
                if current_section == "recommended_policies":
                    # Parse policy (prob%)
                    parts = line[2:].rsplit(' (', 1)
                    if len(parts) == 2:
                        policy = parts[0]
                        prob = parts[1].rstrip('%)')
                        sections[current_section].append((policy, prob))
                    else:
                        sections[current_section].append(line[2:])
                else:
                    sections[current_section].append(line[2:])
            elif current_section == "closing" and line:
                sections["closing"] += line + " "

        sections["closing"] = sections["closing"].strip()

        return sections
    except:
        return {
            "health_assessment": ["InsureBot is currently unavailable. Try later."],
            "insurance_guidance": [],
            "recommended_policies": [],
            "closing": ""
        }


def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if "email" not in session:
            return redirect("/login")
        return f(*a, **k)
    return wrapper


@app.route("/")
def root():
    return redirect("/home")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method=="POST":
        session["email"]=request.form["email"]
        send_otp(session["email"])
        return redirect("/verify")
    return render_template("signup.html")


@app.route("/verify", methods=["GET","POST"])
def verify():
    otp = None
    if app.debug and "email" in session:
        otp = otp_store.get(session["email"])
    if request.method=="POST":
        if verify_otp(session["email"], request.form["otp"]):
            save_user(session["email"])
            return redirect("/login")
        return "❌ Incorrect OTP"
    return render_template("verify.html", otp=otp)


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        session["email"]=request.form["email"]
        return redirect("/index")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/home")


@app.route("/index")
@login_required
def index():
    return render_template("index.html")


@app.route("/history")
@login_required
def history():
    data=get_user_data(session["email"])
    return render_template("history.html", predictions=data.get("predictions", []))


@app.route("/ask", methods=["GET","POST"])
@login_required
def ask():
    answer = None
    if request.method == "POST":
        question = request.form.get("question","")
        answer = gemini_support(question, session.get("personal_details", {}))
    return render_template("ask.html", answer=answer, assistant_name=ASSISTANT_NAME)


@app.route("/support")
def support():
    return render_template("support.html", assistant_name=ASSISTANT_NAME)


@app.route("/support_api", methods=["POST"])
def support_api():
    message = request.get_json().get("message","")
    reply = gemini_support(message, session.get("personal_details", {}))
    return jsonify({"reply": reply, "assistant_name": ASSISTANT_NAME})


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    d=request.form
    region = {"southwest":0,"southeast":1,"northwest":2,"northeast":3}[d["region"].lower()]
    smoker = 1 if d["smoker"].lower()=="yes" else 0
    sex = 1 if d["sex"].lower()=="male" else 0

    df=pd.DataFrame([{
        "age": int(d["age"]),
        "sex": sex,
        "bmi": float(d["bmi"]),
        "children": int(d["children"]),
        "smoker": smoker,
        "region": region
    }])

    session["personal_details"]={
        "name":d["name"],
        "address":d["address"],
        "blood_group":d["blood_group"]
    }

    premium=float(premium_model.predict(df)[0])
    claim_prob=float(claim_model.predict_proba(df[["age","bmi","smoker","region","children"]])[0][1]*100)
    claim=claim_prob>50
    monthly=round(premium/12,2)

    probs = policy_model.predict_proba(df[["age","bmi","smoker","children"]])[0]
    names = list(policy_label_encoder.classes_)
    policies=[(names[i],round(float(p)*100,2)) for i,p in enumerate(probs)]

    data={
        "premium":round(premium,2),
        "monthly_premium":monthly,
        "claim":claim,
        "probability":round(claim_prob,2),
        "policies":policies,
        "personal_details":session["personal_details"],
        "user_data":{
            "age":int(d["age"]),
            "bmi":float(d["bmi"]),
            "smoker":smoker,
            "region":region,
            "children":int(d["children"]),
            "sex":sex
        }
    }

    data["advice"]=gemini_advice(data)
    save_prediction_data(session["email"],data)
    session["last_prediction"]=data

    return render_template("result.html", **data)


@app.route("/download_pdf")
@login_required
def download_pdf():
    data=session.get("last_prediction")
    if not data:
        return redirect("/history")

    filename = f"{data['personal_details']['name'].replace(' ','_')}_Insurance_Report.pdf"
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer)
    styles=getSampleStyleSheet()

    story=[
        Paragraph("Insurance Report",styles["Title"]),
        Spacer(1,12),
        Paragraph(f"Name: {data['personal_details']['name']}",styles["Normal"]),
        Paragraph(f"Blood Group: {data['personal_details']['blood_group']}",styles["Normal"]),
        Paragraph(f"Address: {data['personal_details']['address']}",styles["Normal"]),
        Spacer(1,12),
        Paragraph(f"Premium (Yearly): ₹{data['premium']}",styles["Normal"]),
        Paragraph(f"Monthly Premium: ₹{data['monthly_premium']}",styles["Normal"]),
        Paragraph(f"Claim Probability: {data['probability']}%",styles["Normal"]),
        Spacer(1,12),
        Paragraph("Personalized Advice:",styles["Heading2"]),
    ]

    advice = data.get("advice", {})
    if advice.get("health_assessment"):
        story.append(Paragraph("Health Assessment:", styles["Heading3"]))
        for point in advice["health_assessment"]:
            story.append(Paragraph(f"• {point}", styles["Normal"]))
        story.append(Spacer(1,6))

    if advice.get("insurance_guidance"):
        story.append(Paragraph("Insurance Guidance:", styles["Heading3"]))
        for point in advice["insurance_guidance"]:
            story.append(Paragraph(f"• {point}", styles["Normal"]))
        story.append(Spacer(1,6))

    if advice.get("recommended_policies"):
        story.append(Paragraph("Recommended Policies:", styles["Heading3"]))
        for policy, prob in advice["recommended_policies"]:
            story.append(Paragraph(f"• {policy} ({prob}%)", styles["Normal"]))
        story.append(Spacer(1,6))

    if advice.get("closing"):
        story.append(Paragraph(advice["closing"], styles["Normal"]))

    doc.build(story)
    buffer.seek(0)

    return send_file(buffer, download_name=filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
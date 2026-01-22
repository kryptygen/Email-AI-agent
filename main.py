import os
import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from PyPDF2 import PdfReader
import docx
import csv
import json
import tempfile
from PIL import Image
import pytesseract

load_dotenv()

# -------------------- LLM SETUP --------------------

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    model="tngtech/deepseek-r1t-chimera:free",
    temperature=0.5
)

template = """
Here's a document. Try to answer what the user asked using whatever info you find inside.
Keep the response concise and professional.

Document:
\"\"\"
{document}
\"\"\"

User question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["document", "question"],
    template=template
)

chain = prompt | llm | StrOutputParser()

# -------------------- FILE PARSERS --------------------

def pdf(path):
    try:
        return "\n".join(
            pg.extract_text()
            for pg in PdfReader(path).pages
            if pg.extract_text()
        )
    except Exception:
        return "[Error reading PDF]"

def docxs(path):
    try:
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    except Exception:
        return "[Error reading DOCX]"

def txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "[Error reading TXT]"

def csvx(path):
    try:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                rows.append(", ".join(row))
        return "\n".join(rows)
    except Exception:
        return "[Error reading CSV]"

def jsonx(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.dumps(json.load(f), indent=2)
    except Exception:
        return "[Error reading JSON]"

def image(path):
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception:
        return "[Error reading image]"

# -------------------- EMAIL FETCH --------------------

def fetch():
    mail = imaplib.IMAP4_SSL(os.environ["IMAP_SERVER"])
    mail.login(os.environ["EMAIL_ADDRESS"], os.environ["EMAIL_PASSWORD"])
    mail.select("inbox")

    _, messages = mail.search(None, "UNSEEN")
    if not messages[0]:
        return None, None, None, None

    email_id = messages[0].split()[-1]
    _, msg_data = mail.fetch(email_id, "(RFC822)")
    msg = email.message_from_bytes(msg_data[0][1])

    sender = email.utils.parseaddr(msg["From"])[1]
    subject = msg["Subject"]
    body = ""
    path = None

    for part in msg.walk():
        content_type = part.get_content_type()
        disposition = str(part.get("Content-Disposition"))

        if content_type == "text/plain" and "attachment" not in disposition:
            body = part.get_payload(decode=True).decode(errors="ignore")

        elif "attachment" in disposition:
            filename = part.get_filename()
            if filename and filename.lower().endswith(
                (".pdf", ".docx", ".txt", ".csv", ".json", ".jpg", ".jpeg", ".png")
            ):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as tmp:
                    tmp.write(part.get_payload(decode=True))
                    path = tmp.name

    return sender, subject, body, path

# -------------------- RESPONSE GENERATION --------------------

def response(subject, query, path):
    query = (query or "").strip()

    if not path and not query:
        return """Dear Sir/Madam,

Thank you for reaching out. We could not find any message content or attachment.

Please resend your query or attach a supported document.

Best regards,
Kevin Armaan
"""

    if not path:
        return llm.invoke(query).content

    ext = os.path.splitext(path)[-1].lower()

    content = {
        ".pdf": pdf,
        ".docx": docxs,
        ".txt": txt,
        ".csv": csvx,
        ".json": jsonx,
        ".jpg": image,
        ".jpeg": image,
        ".png": image
    }.get(ext, lambda _: None)(path)

    if not content:
        return """Dear Sir/Madam,

The attached file format is not supported.

Supported formats: PDF, DOCX, TXT, CSV, JSON, JPG, PNG

Best regards,
Kevin Armaan
"""

    if len(content) > 20000:
        content = content[:20000] + "\n\n[Content truncated]"

    if not query:
        query = "Summarize this document."

    result = chain.invoke({
        "document": content,
        "question": query
    })

    return f"""Dear Sir/Madam,

Thank you for your email. Based on the attached document, here is the response:

{result}

If you have further questions, feel free to reply.

Best regards,
Kevin Armaan
"""

# -------------------- SEND EMAIL --------------------

def send(to, subject, body):
    msg = MIMEText(body)
    msg["From"] = os.environ["EMAIL_ADDRESS"]
    msg["To"] = to
    msg["Subject"] = f"Re: {subject}"

    with smtplib.SMTP_SSL(os.environ["SMTP_SERVER"], 465) as server:
        server.login(os.environ["EMAIL_ADDRESS"], os.environ["EMAIL_PASSWORD"])
        server.send_message(msg)

# -------------------- MAIN LOOP --------------------

def main():
    sender, subject, body, path = fetch()

    if sender and not sender.startswith("no-reply@"):
        reply = response(subject, body, path)
        send(sender, subject, reply)
        if path:
            os.remove(path)
        print("Email processed and replied.")
    else:
        print("No new email.")

if __name__ == "__main__":
    main()

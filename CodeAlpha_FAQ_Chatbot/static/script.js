const chat = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("messageInput");
const statusDot = document.getElementById("statusDot");

function addMessage(text, sender, meta) {
  const el = document.createElement("div");
  el.className = `msg ${sender}`;
  if (meta && meta.fallback) el.classList.add("fallback");

  const p = document.createElement("p");
  p.textContent = text;
  el.appendChild(p);

  if (meta && meta.matchedQuestion) {
    const span = document.createElement("span");
    span.className = "meta";
    span.textContent = `Matched: “${meta.matchedQuestion}” (confidence ${meta.confidence})`;
    el.appendChild(span);
  }

  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function showTyping() {
  const el = document.createElement("div");
  el.className = "typing";
  el.id = "typingIndicator";
  el.innerHTML = "<span></span><span></span><span></span>";
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      statusDot.classList.remove("offline");
      statusDot.title = "Connected";
    } else {
      throw new Error("bad status");
    }
  } catch {
    statusDot.classList.add("offline");
    statusDot.title = "Server unreachable";
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage(message, "user");
  input.value = "";
  input.disabled = true;
  showTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    hideTyping();

    if (!res.ok) {
      addMessage("Something went wrong on the server. Please try again.", "bot", { fallback: true });
    } else {
      addMessage(data.reply, "bot", {
        matchedQuestion: data.matched_question,
        confidence: data.confidence,
        fallback: !data.matched_question,
      });
    }
  } catch (err) {
    hideTyping();
    addMessage("Couldn't reach the server. Is app.py running?", "bot", { fallback: true });
    statusDot.classList.add("offline");
  } finally {
    input.disabled = false;
    input.focus();
  }
});

checkHealth();

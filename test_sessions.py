import json, asyncio, httpx
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

BASE_URL = "http://localhost:8000"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# 4 users with distinct emotional profiles
# each has 20 sessions, each session has 10+ turns
# turns alternate: user_text → llm_response (llm responses not classified)

USERS = {
    "user_001": {
        "name": "Samuel",
        "profile": "Young professional, grief after losing father, work stress",
        "sessions": [
            # SESSION 1 — early grief
            [
                ("I keep going to call him and then I remember", "That reflex of reaching for someone who's gone is one of the hardest parts of grief. Your mind hasn't caught up yet. When did you lose him?"),
                ("Three months ago. It still doesn't feel real", "Three months is still very raw. Grief doesn't follow a schedule. What was he like?"),
                ("He was the kind of man who showed up. Always.", "That kind of presence leaves a very specific absence. You notice it in small moments."),
                ("I drove past the hospital yesterday and couldn't look", "Your body is protecting you. That instinct to look away isn't weakness — it's self-preservation."),
                ("I don't know how to be okay with any of this", "You don't have to be okay with it. You just have to get through today. What's one thing that helped you get out of bed this morning?"),
                ("Work, I guess. It gives me something to do with my hands", "Structure can be a lifeline early in grief. The body keeps moving even when the mind is elsewhere."),
                ("My colleagues don't know. I haven't told anyone at work", "Carrying that quietly takes a lot of energy. Is there anyone there you trust enough to tell?"),
                ("Maybe one person. But I don't want to make it weird", "That's a reasonable concern. You don't have to announce it — even one person knowing can lighten the load."),
                ("I'll think about it. I just don't want their pity", "That makes sense. You want acknowledgment, not performance. There's a real difference."),
                ("Yeah. Exactly that.", "You named it well. Holding onto that distinction might help you decide who's safe to tell."),
            ],
            # SESSION 2 — work pressure emerging
            [
                ("My manager gave my project to someone else today", "That stings, especially when you're already stretched thin. Did they give you a reason?"),
                ("They said I seemed distracted. They're not wrong but still", "Being seen clearly and still feeling the sting of the consequence — that's a complicated feeling."),
                ("I wanted to argue but I just sat there", "Sometimes silence is all we have when we're running on empty. What did you feel sitting there?"),
                ("Like I was watching myself from outside", "Dissociation often shows up when we're overwhelmed. Your mind stepped back to protect you."),
                ("Is that normal", "Very. It's a common stress response, especially when grief and pressure overlap. How's your sleep?"),
                ("I sleep but I don't rest", "Unrestorative sleep is exhausting in a specific way — you wake up already tired. How long has that been?"),
                ("Since he died honestly", "Your body is still in alert mode. That's grief living in the nervous system, not just the mind."),
                ("I didn't know grief could feel physical", "It very much can. Chest tightness, fatigue, that hollow feeling — all of it is real."),
                ("I have all of those", "Thank you for telling me that. These aren't things to push through alone. Are you eating regularly?"),
                ("Not really. I forget sometimes", "When the body forgets hunger, it's usually a sign the nervous system is overwhelmed. Small meals still count."),
                ("I'll try", "That's enough. One small thing at a time."),
            ],
            # SESSION 3 — small moment of relief
            [
                ("I told my colleague today. The one I mentioned", "How did it go?"),
                ("Better than I expected. She lost her mum two years ago", "Sometimes the right person already knows the terrain. How did it feel to say it out loud?"),
                ("Like putting something heavy down for a second", "That image says a lot. You've been carrying it alone for a while."),
                ("She didn't make it weird. She just said she was sorry and meant it", "That kind of response is rare and it matters. Do you feel a bit less alone at work now?"),
                ("A little. Yeah", "A little is real. Don't discount it."),
                ("I even ate lunch today. Properly", "That's not small. That's your body being taken care of. Well done."),
                ("She sat with me. I think that helped", "Shared presence without pressure — that's exactly what grief needs sometimes."),
                ("I still missed him at lunch though. We used to call during my lunch break", "Of course you did. Love doesn't disappear just because the person does. Those moments will keep coming."),
                ("Will they always hurt this much", "The intensity changes over time, even when it doesn't feel like it will. The love stays — the pain gradually finds somewhere to live alongside it."),
                ("That's a strange way to put it but it makes sense", "Grief is strange. It deserves strange language sometimes."),
            ],
            # SESSION 4 — anger surfaces
            [
                ("I'm angry today. I don't know why", "Anger often shows up in grief when the sadness gets too heavy. It doesn't need a reason. What does it feel like?"),
                ("Like I want to break something", "That's a very honest answer. Have you felt this before in the past few months?"),
                ("No. Usually I just feel sad or numb", "Anger can feel alarming after mostly quiet grief. But it's part of the same thing. What happened today?"),
                ("Nothing specific. I just woke up furious", "Sometimes the body holds it until it's ready to release it. Waking up angry often means something was processed overnight."),
                ("He didn't even get to retire. He worked his whole life", "There it is. That's a real injustice and you're allowed to be furious about it."),
                ("It's not fair", "No. It isn't. You don't have to make peace with that today."),
                ("Everyone keeps saying he's in a better place", "People say that because they don't know what else to say. It doesn't mean you have to find it comforting."),
                ("It makes me want to scream", "Then scream. Not at people — but somewhere safe. That energy needs somewhere to go."),
                ("Maybe I'll go running later", "That's a good instinct. Movement is one of the few things that actually metabolises anger."),
                ("I haven't run since he died", "Then today might be the right day to start again. Let us know how it goes."),
            ],
            # SESSION 5 — plateau, routine returning
            [
                ("I went running. Three times this week", "Three times. That's not a small thing — that's a pattern forming."),
                ("It helps. I don't think while I'm running", "The body taking over the mind for a while — that's exactly what you needed."),
                ("Work is still hard but I'm managing", "Managing is real progress. What does managing look like for you right now?"),
                ("Showing up. Doing the work. Not spiralling when it's quiet", "Not spiralling in the quiet is significant. That used to be when it hit hardest. What changed?"),
                ("I think the running gives me something to do with the feeling", "You built yourself a container for it. That's actually a healthy coping mechanism."),
                ("I didn't plan it that way", "Most effective coping doesn't get planned. The body finds what it needs."),
                ("I still miss him every day", "You always will. That doesn't go away. But it sounds like you're learning to carry it differently."),
                ("Is that what healing looks like", "For a lot of people, yes. Not feeling less — just holding it more steadily."),
                ("I thought healing meant not feeling it anymore", "That's a common myth. Healing usually means integrating it — the grief becomes part of you rather than all of you."),
                ("That actually helps to hear", "Good. You're doing harder work than you realise."),
            ],
            # SESSIONS 6–20 abbreviated as realistic continuations
            *[
                [
                    (f"Things have been {'up' if i % 3 == 0 else 'hard' if i % 3 == 1 else 'mixed'} lately", "Tell me more about that."),
                    ("I had a moment where I felt okay today and then felt guilty about it", "Grief guilt is very common — feeling okay can feel like betrayal. It isn't."),
                    ("Why does it feel like I'm forgetting him when I feel okay", "Feeling okay doesn't erase him. It means you're still here, and that's what he'd want."),
                    ("I know logically but it doesn't feel that way", "Logic and grief rarely agree. Both things can be true at the same time."),
                    ("I've been talking to his old friends. It helps to hear stories", "Keeping his memory alive through others — that's a healthy way to stay connected."),
                    ("One of them told me something about him I never knew", "Those discoveries can feel like small gifts. What was it like to hear it?"),
                    ("Strange. Good. Like meeting a part of him I missed", "That's a beautiful way to put it. He's still being revealed to you."),
                    ("I didn't expect grief to feel like that", "Grief has more textures than we expect. You're finding them one at a time."),
                    ("Some days are still really dark", "Those days are part of it too. How do you get through the dark ones now?"),
                    ("Running. Calling my colleague. Sometimes just sleeping", "You have a toolkit now. That's different from three months ago."),
                ]
                for i in range(15)
            ]
        ]
    },

    "user_002": {
        "name": "Aisha",
        "profile": "University student, anxiety, relationship stress, faith tension",
        "sessions": [
            # SESSION 1 — exam anxiety
            [
                ("I can't sleep. Exams are in two weeks and I feel completely blank", "That blankness under pressure is really common — the mind sometimes goes quiet right when you need it most. How long has the sleep been disrupted?"),
                ("About a week. I lie there and just think about everything I haven't done", "The loop of unfinished tasks playing on repeat at night is exhausting. Is there one exam you're most afraid of?"),
                ("Statistics. I don't understand it and I never have", "That specific fear is actually more manageable than general dread. What part of statistics feels most foreign?"),
                ("All of it honestly. I sit in the lecture and understand nothing", "That feeling of sitting in a room where nothing lands — that's really isolating. Have you talked to the lecturer?"),
                ("No. I'm too embarrassed", "The embarrassment makes sense but the lecturer has seen this many times. You wouldn't be surprising them."),
                ("What if they think I should already know this", "They might think you needed help earlier — and that's okay. Asking now is still better than not asking."),
                ("I typed an email to them and deleted it three times", "Three drafts means you want to send it. What stops you at the last moment?"),
                ("Fear that they'll confirm I'm stupid", "That fear is the exam anxiety talking, not reality. Can we try drafting that email together right now?"),
                ("Okay. Maybe. Yes", "Let's start with just one sentence — what do you need from them?"),
                ("I need them to explain it like I've never heard it before", "That's a perfect first sentence. Clear and honest. You're not stupid — you're someone who needs a different entry point."),
            ],
            # SESSION 2 — relationship pressure
            [
                ("My boyfriend said I've been distant. He's not wrong", "Anxiety has a way of pulling us inward. How did that conversation go?"),
                ("He wasn't angry. Just sad. Which is somehow worse", "Sadness from someone you love often hits harder than anger. It implies they miss you."),
                ("I miss him too but I can't be present right now", "You're stretched beyond capacity. That's not absence of love — it's absence of bandwidth."),
                ("He doesn't understand that distinction", "That's a hard gap to bridge when someone is hurting. Have you been able to explain what's happening for you?"),
                ("I tried but I started crying and then he panicked and then I had to comfort him", "You ended up managing his response to your pain. That's an exhausting reversal."),
                ("That happens a lot with us actually", "That pattern — where your vulnerability becomes something you manage for him — is worth paying attention to."),
                ("I never thought of it that way", "It doesn't mean he's a bad person. But it does mean something about the dynamic that might be worth exploring."),
                ("I love him but I'm also tired in a way I can't explain", "Both those things can be true. Love and exhaustion aren't opposites."),
                ("What do I do", "You don't have to decide anything today. But naming the pattern is the first step."),
                ("I've never named it before", "Now you have. That's not nothing."),
            ],
            # SESSION 3 — faith conflict
            [
                ("I stopped praying this week and I feel guilty about it", "Guilt around spiritual practice often means it matters to you deeply. What made you stop?"),
                ("I don't know if God is listening. Exams, relationship, everything — I just felt alone", "That feeling of silence from something you've leaned on — that's a specific kind of loneliness."),
                ("My mum would be devastated if she knew", "You're holding her expectation alongside your own doubt. That's a heavy double weight."),
                ("She prays every morning. It's just part of her", "Faith that's woven into someone's daily rhythm can look effortless from the outside. But hers was built over decades."),
                ("Maybe I'm just weak", "Doubt isn't weakness. It's often the beginning of a more honest faith, not the end of faith altogether."),
                ("I want to believe that", "The wanting is still there. That matters."),
                ("I opened my Bible last night but couldn't read it", "You showed up. Even sitting with it is something."),
                ("I just stared at the page", "Sometimes that's all there is. Presence without comprehension is still presence."),
                ("Do other people feel this", "Many people do. The ones who stay in faith often have a period where it went very quiet."),
                ("That actually helps. I thought it was just me", "It's rarely just you. That's true of most things you're ashamed of."),
            ],
            # SESSIONS 4–20 abbreviated
            *[
                [
                    ("I had a better week", "Tell me what made it better."),
                    ("I sent the email to my statistics lecturer", "You sent it. How did they respond?"),
                    ("They were actually really kind. Set up a tutorial session", "Of course they were. You asked clearly and they responded to that."),
                    ("I went to the tutorial and understood three things I never understood before", "Three things. That's a foundation. How did it feel to understand them?"),
                    ("Like a door opening a little", "That image is exactly right. One crack of light changes everything."),
                    ("My boyfriend and I talked properly. I told him about the pattern", "How did he receive that?"),
                    ("He was quiet for a long time and then said he didn't realise", "Quiet first often means real thinking. That's a better sign than immediate defence."),
                    ("We're okay I think. Better than before actually", "Naming the thing changed the thing. That happens sometimes."),
                    ("I prayed this morning. It felt strange but I did it", "Strange and done is still done. How did you feel after?"),
                    ("A little lighter. Like I put something down", "That's worth holding onto. Not as proof — just as data about what helps you."),
                ]
                for _ in range(17)
            ]
        ]
    },

    "user_003": {
        "name": "Brian",
        "profile": "Mid-career professional, burnout, suppressed anger, numbness",
        "sessions": [
            # SESSION 1 — presenting as fine
            [
                ("I'm fine. My wife said I should try this", "Welcome. What did your wife notice that made her suggest it?"),
                ("She says I've been unreachable. Her word", "Unreachable is a specific word. What do you think she means by it?"),
                ("I'm home every evening. I don't know what she means", "Being physically present and emotionally available aren't always the same thing. Does that distinction resonate?"),
                ("Maybe. I'm tired when I get home", "Tired in what way — body or something else?"),
                ("Both. I sit down and I just can't engage", "That kind of depletion where engagement feels impossible — how long has it been like that?"),
                ("A year. Maybe more", "A year of that is significant. What does work look like right now?"),
                ("I manage a team of twelve. It never stops", "Twelve people, constant demand, nothing stopping — that's a very full container. What happens when it overflows?"),
                ("It doesn't. I just keep going", "That's the thing worth examining. What does keeping going cost you?"),
                ("I don't know. I don't think about it", "That might be part of what your wife is noticing. The not-thinking can look like absence."),
                ("I hadn't considered that", "You're here. That's already a form of paying attention."),
            ],
            # SESSION 2 — burnout admitted
            [
                ("She was right. I think I'm burnt out", "Saying that is harder than it sounds. What made you change your mind?"),
                ("I sat in a meeting yesterday and felt nothing. Completely blank. I used to care about this work", "The contrast between who you were and who you are in that room — that's grief of a kind."),
                ("I didn't think of it as grief", "Burnout often carries grief inside it. You're mourning a version of yourself that had energy and investment."),
                ("That version feels very far away", "How long ago do you think it started disappearing?"),
                ("Slowly. Over about two years I think", "Gradual erosion is harder to spot than sudden collapse. You don't notice until there's very little left."),
                ("My team is suffering because of it. I can see it", "Awareness of impact is painful but it means you're still engaged on some level."),
                ("I snap at them sometimes. I hate that", "The snapping is usually the overflow point — it means you've been holding too much for too long."),
                ("I apologise after. But I keep doing it", "Repeated apology without change is its own exhaustion. What would need to change to reduce the pressure?"),
                ("I don't know. Maybe headcount. Maybe expectations", "Are those things you can influence?"),
                ("Possibly. I haven't tried. I've just absorbed it", "Absorbing without advocating is a pattern. It makes sense — but it's also the pattern that's eroding you."),
            ],
            # SESSION 3 — anger finally admitted
            [
                ("I'm angry. I realised it this week", "Tell me about that realisation."),
                ("I've been angry for years. I just called it something else. Stress. Tiredness.", "That reframe is significant. Anger dressed as tiredness is still anger — it just doesn't get addressed."),
                ("It's safer to be tired than angry in my industry", "That's a real observation. What would happen if the anger were visible?"),
                ("People would see me as unstable. Difficult.", "So you perform tiredness and suppress anger. That performance has a cost."),
                ("A very high one apparently", "What does the anger actually want?"),
                ("To be heard. To matter. To not be the one who always absorbs everything", "Those are legitimate needs, not character flaws. How long have you been waiting for them to be met?"),
                ("A long time. Maybe my whole career", "That's a heavy wait. Is there anyone at work who knows this version of you?"),
                ("No. I keep it very clean at work", "Clean surfaces, hidden cost. What about at home?"),
                ("My wife gets the edges of it. That's not fair on her", "You're aware of the unfairness. That awareness is important. What would it look like to be more honest with her?"),
                ("I don't know where to start", "You could start with exactly what you just told me — that you've been angry for years and only just named it."),
            ],
            # SESSIONS 4–20 abbreviated
            *[
                [
                    ("I talked to my wife", "How did it go?"),
                    ("She cried. Said she'd been waiting for me to come back", "Come back — that's the word she used. What did it mean to you?"),
                    ("That I've been gone longer than I thought", "That lands heavily. What did you feel hearing it?"),
                    ("Guilt. But also something like relief", "Relief that she still wants you there. Guilt that it took this long. Both make sense."),
                    ("I told her about the anger. She wasn't surprised", "Partners often sense what we hide. Her not being surprised might be reassuring in a strange way."),
                    ("She said she'd rather have the angry version than the absent version", "That's a significant thing to say. She's choosing real over managed."),
                    ("I'm trying to be less managed", "What does less managed look like in practice?"),
                    ("Saying when something bothers me before it builds", "That's the whole intervention, really. Before it builds is everything."),
                    ("It's hard. I have twenty years of holding it in", "Twenty years of muscle memory. It won't unlearn overnight. But you're practising."),
                    ("I am. I really am", "That's all it takes to start."),
                ]
                for _ in range(17)
            ]
        ]
    },

    "user_004": {
        "name": "Zara",
        "profile": "Recent relocation, loneliness, identity confusion, tentative optimism",
        "sessions": [
            # SESSION 1 — new city, alone
            [
                ("I moved to a new city six weeks ago and I don't know anyone", "Six weeks is still very early. How has it felt day to day?"),
                ("Quiet in a way I wasn't prepared for", "That specific quiet — the absence of familiar noise — can be louder than actual noise."),
                ("I keep the TV on just to have voices", "That instinct makes complete sense. Sound as company is a real thing."),
                ("Is that strange", "Not at all. It's a practical response to an absence."),
                ("I thought I'd be more okay with it. I'm usually independent", "Independence and loneliness aren't opposites. Even very independent people need some connection."),
                ("I never needed people before. Or I thought I didn't", "Sometimes it takes their absence to discover how much they were there."),
                ("I've been video calling home every night", "Every night suggests a real need. How does it feel after the calls?"),
                ("Better for an hour and then worse", "The contrast between connection and its ending can be sharper than before the call. That's common with homesickness."),
                ("Is this homesickness? I thought I was past that", "Homesickness doesn't have an age limit or a rational threshold. You miss what you miss."),
                ("My mum said give it six months", "Six months is reasonable advice. You're six weeks in. You're still in the hardest part."),
            ],
            # SESSION 2 — identity shift
            [
                ("I don't know who I am here. That sounds dramatic", "It doesn't sound dramatic at all. Identity is partly built from context — people who know you, places that hold your history."),
                ("Back home I knew exactly who I was", "And here that scaffolding is gone. You're you without the familiar mirrors."),
                ("The mirrors. That's exactly it", "Without people who reflect back a familiar version of you, it can feel like you've become vague."),
                ("Vague. Yes. I feel vague", "That vagueness is disorienting but it's also an opening — who do you want to be in this new context?"),
                ("I haven't thought about it that way", "Most people don't get to rebuild from a blank page. It's uncomfortable but it's rare."),
                ("I'm not sure I wanted a blank page", "That makes sense. It wasn't chosen — it was circumstantial. There's grief in that."),
                ("I moved for a good reason. Career opportunity. I keep telling myself that", "The reason being good doesn't cancel the loss. Both are true."),
                ("I feel guilty for being sad about it", "Grief guilt again — feeling you don't have permission to be sad because you chose it."),
                ("Exactly that", "You have permission. Good decisions can still hurt."),
                ("Nobody told me that", "Now someone has."),
            ],
            # SESSION 3 — first connection
            [
                ("I met someone at a coffee shop. We talked for an hour", "An hour is a real conversation. How did it feel?"),
                ("Like remembering something I'd forgotten", "That image — memory of a feeling — says a lot about how long you've been without it."),
                ("I didn't even get her number. I'm annoyed at myself", "What held you back?"),
                ("Fear I think. What if she didn't want to", "Better to have asked and been told no than to carry the wondering. But the fear makes sense."),
                ("She mentioned a bookshop nearby. I went the next day hoping to see her", "You went back. That's not nothing — that's hope acting on itself."),
                ("She wasn't there. But the bookshop was nice", "You found something while looking for something else. That happens."),
                ("I bought three books. First time I've bought anything here that wasn't groceries", "Claiming a space — even a small one — is the beginning of belonging somewhere."),
                ("I hadn't thought of it as claiming space", "Every small root you put down matters. A bookshop you like is a root."),
                ("I went back again yesterday", "Twice. You're building a place."),
                ("Maybe. It's early days but maybe", "Maybe is enough."),
            ],
            # SESSIONS 4–20 abbreviated
            *[
                [
                    ("I joined a reading group at that bookshop", "You went back and you stayed. How is it?"),
                    ("Strange at first. But I went twice and it's getting easier", "Twice is the threshold where strange starts becoming familiar."),
                    ("There's a woman there who reminds me of my best friend back home", "That familiarity can be a bridge — let it be one without expecting her to be the same person."),
                    ("I know. I'm trying not to project", "The awareness is the protection. You'll be okay."),
                    ("Work is good actually. Better than I expected", "What's working about it?"),
                    ("The problems are interesting. I forget myself in the work", "Absorption in interesting problems is one of the better forms of relief."),
                    ("I still call home every night but it doesn't hurt as much after", "The landing is getting softer. That's progress."),
                    ("I caught myself smiling on the bus today for no reason", "No reason is usually a reason. Something is settling."),
                    ("Maybe I'm going to be okay here", "You already are, a little. It's just hard to see it from inside it."),
                    ("I think I needed to hear that", "You needed to know it first. Hearing it just confirmed it."),
                ]
                for _ in range(17)
            ]
        ]
    }
}

async def classify_turn(client: httpx.AsyncClient, text: str, session_id: str) -> dict:
    resp = await client.post(f"{BASE_URL}/classify", json={
        "text": text,
        "session_id": session_id
    })
    resp.raise_for_status()
    return resp.json()

async def create_session(client: httpx.AsyncClient, user_id: str) -> str:
    resp = await client.post(f"{BASE_URL}/classify", json={
        "text": "Starting session.",
        "user_id": user_id
    })
    resp.raise_for_status()
    return resp.json()["session_id"]

def compute_emotion_fingerprint(all_turns: list[dict]) -> dict:
    counter = Counter()
    confidence_totals = defaultdict(float)
    for turn in all_turns:
        for e in turn["top_3"]:
            counter[e["emotion"]] += 1
            confidence_totals[e["emotion"]] += e["confidence"]
    total = sum(counter.values()) or 1
    return {
        emotion: {
            "frequency": count,
            "frequency_pct": round(count / total * 100, 1),
            "avg_confidence": round(confidence_totals[emotion] / count, 3)
        }
        for emotion, count in counter.most_common()
    }

def compute_emotion_arc(sessions: list[dict]) -> list[dict]:
    arc = []
    for s in sessions:
        if not s["turns"]:
            continue
        dominant = Counter()
        for t in s["turns"]:
            if t["top_3"]:
                dominant[t["top_3"][0]["emotion"]] += 1
        top = dominant.most_common(1)[0][0] if dominant else "neutral"
        arc.append({
            "session_index": s["session_index"],
            "dominant_emotion": top,
            "turn_count": len(s["turns"]),
        })
    return arc

async def run_user(user_id: str, user_data: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"User: {user_data['name']} ({user_id})")
    print(f"Profile: {user_data['profile']}")
    print(f"{'='*60}")

    user_report = {
        "user_id": user_id,
        "name": user_data["name"],
        "profile": user_data["profile"],
        "sessions": [],
        "emotion_fingerprint": {},
        "emotion_arc": [],
        "summary": {}
    }

    all_classified_turns = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for session_idx, session_turns in enumerate(user_data["sessions"]):
            print(f"\n  Session {session_idx + 1}/{len(user_data['sessions'])}")

            # create session with first user turn
            first_user_text = session_turns[0][0]
            init_resp = await client.post(f"{BASE_URL}/classify", json={
                "text": first_user_text,
                "user_id": user_id
            })
            init_resp.raise_for_status()
            init_data = init_resp.json()
            session_id = init_data["session_id"]

            session_report = {
                "session_index": session_idx + 1,
                "session_id": session_id,
                "turns": [],
                "dominant_emotion": None,
                "session_summary": {}
            }

            # log first turn
            turn_report = {
                "turn": 1,
                "role": "user",
                "text": first_user_text,
                "llm_response": session_turns[0][1],
                "top_3": init_data["top_3"],
                "reasoning": init_data.get("reasoning", ""),
                "translation": init_data.get("translation", "")
            }
            session_report["turns"].append(turn_report)
            all_classified_turns.append(init_data)
            print(f"    Turn 1: {init_data['top_3'][0]['emotion']} ({init_data['top_3'][0]['confidence']:.2f}) — {first_user_text[:50]}...")

            # remaining turns
            for turn_idx, (user_text, llm_response) in enumerate(session_turns[1:], start=2):
                # classify user turn with session context (server uses last 4 turns via Redis)
                try:
                    resp = await client.post(f"{BASE_URL}/classify", json={
                        "text": user_text,
                        "session_id": session_id
                    })
                    resp.raise_for_status()
                except Exception as e:
                    print(f"    ERROR on turn {turn_idx}: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"    Response text: {e.response.text}")
                    raise
                turn_data = resp.json()

                turn_report = {
                    "turn": turn_idx,
                    "role": "user",
                    "text": user_text,
                    "llm_response": llm_response,
                    "top_3": turn_data["top_3"],
                    "reasoning": turn_data.get("reasoning", ""),
                    "translation": turn_data.get("translation", "")
                }
                session_report["turns"].append(turn_report)
                all_classified_turns.append(turn_data)

                top = turn_data["top_3"][0]
                print(f"    Turn {turn_idx}: {top['emotion']} ({top['confidence']:.2f}) — {user_text[:50]}...")

                await asyncio.sleep(0.3)  # rate limit buffer

            # per-session dominant emotion
            session_counter = Counter(
                t["top_3"][0]["emotion"]
                for t in session_report["turns"]
                if t["top_3"]
            )
            session_report["dominant_emotion"] = session_counter.most_common(1)[0][0]
            session_report["session_summary"] = {
                "total_turns": len(session_report["turns"]),
                "emotion_distribution": dict(session_counter.most_common()),
            }

            user_report["sessions"].append(session_report)
            print(f"    Dominant: {session_report['dominant_emotion']}")

    # user-level analytics
    user_report["emotion_fingerprint"] = compute_emotion_fingerprint(all_classified_turns)
    user_report["emotion_arc"] = compute_emotion_arc(user_report["sessions"])

    top_emotion = list(user_report["emotion_fingerprint"].keys())[0] if user_report["emotion_fingerprint"] else "neutral"
    user_report["summary"] = {
        "total_sessions": len(user_report["sessions"]),
        "total_turns_classified": len(all_classified_turns),
        "top_emotion_overall": top_emotion,
        "emotion_fingerprint_top5": dict(
            list(user_report["emotion_fingerprint"].items())[:5]
        ),
        "arc_summary": [
            f"Session {a['session_index']}: {a['dominant_emotion']}"
            for a in user_report["emotion_arc"]
        ]
    }

    # save per-user report
    user_file = REPORTS_DIR / f"{user_id}_report.json"
    with open(user_file, "w") as f:
        json.dump(user_report, f, indent=2)
    print(f"\n  Saved: {user_file}")

    return user_report

async def main():
    print(f"EmpathAI Multi-User Test")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Users: {len(USERS)} | Sessions per user: 20 | Min turns per session: 10")

    all_user_reports = []

    for user_id, user_data in USERS.items():
        report = await run_user(user_id, user_data)
        all_user_reports.append(report)

    # master report
    master = {
        "generated_at": datetime.now().isoformat(),
        "total_users": len(all_user_reports),
        "users": {
            r["user_id"]: {
                "name": r["name"],
                "profile": r["profile"],
                "summary": r["summary"],
                "emotion_fingerprint": r["emotion_fingerprint"],
                "emotion_arc": r["emotion_arc"]
            }
            for r in all_user_reports
        },
        "comparative": {
            "by_top_emotion": {
                r["user_id"]: r["summary"]["top_emotion_overall"]
                for r in all_user_reports
            },
            "by_total_turns": {
                r["user_id"]: r["summary"]["total_turns_classified"]
                for r in all_user_reports
            },
            "emotion_fingerprints": {
                r["user_id"]: dict(list(r["emotion_fingerprint"].items())[:5])
                for r in all_user_reports
            }
        }
    }

    master_file = REPORTS_DIR / "master_report.json"
    with open(master_file, "w") as f:
        json.dump(master, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    for r in all_user_reports:
        s = r["summary"]
        print(f"\n{r['name']} ({r['user_id']})")
        print(f"  Sessions : {s['total_sessions']}")
        print(f"  Turns    : {s['total_turns_classified']}")
        print(f"  Top emotion : {s['top_emotion_overall']}")
        print(f"  Fingerprint (top 5):")
        for emotion, data in s["emotion_fingerprint_top5"].items():
            print(f"    {emotion:<18} {data['frequency_pct']}%  avg conf {data['avg_confidence']}")

    print(f"\nMaster report : {master_file}")
    print(f"Per-user files: {REPORTS_DIR}/user_00X_report.json")

if __name__ == "__main__":
    asyncio.run(main())
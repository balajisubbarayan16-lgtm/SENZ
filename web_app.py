"""Mobile-friendly local web interface for SPONGE.

The PC hosts this page and continues to run the camera, face models, Ollama,
Vosk, and MongoDB locally. Phones connect to the PC over the local network.
"""

import json
import re
import secrets
import socket
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from memory_store import (
    answer_meeting_question,
    answer_conversationally,
    connect_database,
    _extract_meeting_fields,
    generate_notification,
    interpret_voice_utterance,
    list_appointments,
    save_memory,
    save_appointment,
    save_profile,
)
from visitor_assistant import (
    MATCH_THRESHOLD,
    PEOPLE_PATH,
    SAMPLES_TO_COLLECT,
    VOICE_MODEL_PATH,
    face_embedding,
    load_models,
    load_people,
    open_camera,
    start_voice_transcriber,
)


ROOT = Path(__file__).resolve().parent
PORT = 8765
MAX_REQUEST_BYTES = 1_000_000


PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <meta name="theme-color" content="#101a2c">
  <title>SPONGE</title>
  <style>
    :root{color-scheme:dark;--bg:#0b1220;--panel:#111c2f;--panel2:#16243a;--line:#263650;--text:#eef4ff;--muted:#a7b5ca;--blue:#7da8ff;--green:#6cdbac;--amber:#ffd27a}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(1000px 500px at 20% -10%,#1a3155 0%,transparent 60%),var(--bg);color:var(--text);font:16px/1.45 Segoe UI,system-ui,sans-serif}
    .wrap{max-width:1120px;margin:auto;padding:22px 16px 40px}.top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:18px}.brand{font-size:23px;font-weight:750;letter-spacing:-.4px}.sub{color:var(--muted);font-size:13px}.badge{border:1px solid #285543;color:var(--green);background:#102a25;padding:7px 11px;border-radius:99px;font-size:11px;font-weight:700;white-space:nowrap}
    .grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,.9fr);gap:14px}.card{background:linear-gradient(160deg,#142139,var(--panel) 70%);border:1px solid var(--line);border-radius:18px;padding:17px;box-shadow:0 12px 32px #0002}.card h2{font-size:13px;letter-spacing:.11em;color:var(--muted);text-transform:uppercase;margin:0 0 12px}.video{width:100%;min-height:220px;max-height:62vh;object-fit:contain;background:#050a13;border-radius:12px;display:block}.camera-head{display:flex;justify-content:space-between;align-items:center;gap:8px}.status{font-size:12px;color:var(--muted)}.status.on{color:var(--green)}
    .person{font-size:25px;font-weight:750;margin:2px 0 3px;overflow-wrap:anywhere}.relationship{color:var(--muted);min-height:24px}.count{margin-top:13px;padding:10px 12px;border-radius:11px;background:#0d1728;color:var(--blue);font-size:13px;font-weight:650}.notice{padding:12px 13px;border-radius:12px;border:1px solid #37465f;background:#0d1728;color:var(--text);margin-top:12px;min-height:50px}.notice.ready{border-color:#2e7259;color:var(--green)}
    .reminder{font-size:17px;min-height:58px}.controls{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px;align-items:center}.controls select,.voice-picker select{border:1px solid var(--line);border-radius:11px;background:var(--panel2);color:var(--text);padding:11px 12px;font:14px Segoe UI,system-ui}.controls select:disabled{opacity:.55}.button{border:1px solid var(--line);border-radius:11px;background:var(--panel2);color:var(--text);padding:11px 14px;font:650 14px Segoe UI,system-ui;cursor:pointer}.button.primary{background:#4d7de0;border-color:#4d7de0;color:white}.button:disabled{opacity:.45;cursor:default}.form{display:flex;gap:8px;margin-top:10px}.form input{min-width:0;flex:1;border:1px solid var(--line);border-radius:11px;background:#0a1424;color:var(--text);padding:12px;font:15px Segoe UI,system-ui}.form input:focus{outline:2px solid #5687ee}.examples{color:var(--muted);font-size:13px;margin-top:12px}.examples b{color:#d7e4fa}.toggle{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px;margin-top:13px}.voice-picker{display:flex;align-items:center;flex-wrap:wrap;gap:8px;color:var(--muted);font-size:13px;margin-top:8px}.foot{color:#8190a7;font-size:12px;margin-top:16px}
    .modal[hidden]{display:none}.modal{position:fixed;inset:0;z-index:10;display:grid;place-items:center;padding:18px;background:#050a13cc}.modal-card{width:min(100%,480px);max-height:90vh;overflow:auto;padding:22px;border:1px solid var(--line);border-radius:18px;background:var(--panel);box-shadow:0 24px 70px #0009}.modal-card h2{margin:0 0 16px;color:var(--text);font-size:18px}.modal-card label{display:grid;gap:6px;margin:12px 0;color:var(--muted);font-size:13px}.modal-card input,.modal-card textarea{width:100%;border:1px solid var(--line);border-radius:10px;background:#0a1424;color:var(--text);padding:12px;font:15px Segoe UI,system-ui}.modal-card textarea{resize:vertical}.modal-actions{display:flex;justify-content:flex-end;gap:9px;margin-top:16px}.appointment-list{border-top:1px solid var(--line);margin-top:18px;padding-top:14px}.appointment-list h3{margin:0 0 8px;font-size:14px;color:var(--muted)}.appointment-list ul{list-style:none;padding:0;margin:0;display:grid;gap:8px}.appointment-list li{padding:10px 12px;border-radius:10px;background:#0a1424;font-size:13px}.appointment-list li span{display:block;color:var(--muted);margin-top:3px}.login{max-width:420px;margin:12vh auto;padding:24px}.login h1{font-size:24px;margin:0 0 6px}.login p{color:var(--muted)}.login .form{flex-direction:column}.login input{width:100%}
    @media(max-width:760px){.wrap{padding:15px 12px 28px}.grid{grid-template-columns:1fr}.video{min-height:190px;max-height:46vh}.person{font-size:22px}.top{align-items:flex-start}.brand{font-size:20px}.card{padding:15px}.controls .button{flex:1}.form .button{flex:0 0 auto}}
  </style>
</head>
<body>
  <main class="wrap">
    <header class="top"><div><div class="brand">SPONGE</div><div class="sub">Private visitor context, hosted on your PC</div></div><div class="badge">● LOCAL NETWORK</div></header>
    <section class="grid">
      <div class="card">
        <div class="camera-head"><h2>Live camera</h2><span id="cameraStatus" class="status">Camera off</span></div>
        <img id="video" class="video" alt="Processed camera preview">
        <video id="phoneCamera" playsinline muted style="display:none"></video>
        <div class="controls"><button id="start" class="button primary" onclick="control('start')">Start on PC</button><button id="startPhone" class="button" onclick="startPhone()">Use phone camera &amp; mic</button><label class="sub" for="phoneFacing">Phone camera</label><select id="phoneFacing" aria-label="Choose phone camera"><option value="environment">Rear camera</option><option value="user">Front camera</option></select><button id="switchPhone" class="button" onclick="switchPhoneCamera()" disabled>Switch camera</button><button id="pause" class="button" onclick="control('pause')">Pause recognition</button><button id="mute" class="button" onclick="control('mute')">Mute speech</button></div>
        <label class="toggle"><input id="speech" type="checkbox" checked> Speak updates on this device</label>
        <label class="voice-picker" for="speechVoice">Speech voice <select id="speechVoice"><option value="auto">Auto (pleasant English voice)</option></select></label>
      </div>
      <div class="card">
        <h2>Person in view</h2><div id="person" class="person">Waiting for camera</div><div id="relationship" class="relationship"></div>
        <div id="count" class="count">Face samples: 0/15</div>
        <button id="editProfile" class="button" type="button" onclick="openEditDialog()" disabled>Edit recognized person</button>
        <button id="openAppointments" class="button" type="button" onclick="openAppointmentDialog()">Appointments</button>
        <div id="notice" class="notice">Start the camera on your PC to begin.</div>
        <h2 style="margin-top:18px">Latest reminder</h2><div id="reminder" class="reminder">No reminder yet.</div>
        <div class="sub" id="voiceStatus" style="margin-top:8px">Choose PC camera/mic or phone camera/mic. Ask a question or tell me something to remember.</div>
        <form class="form" id="commandForm"><input id="command" autocomplete="off" placeholder="Ask me something…"><button class="button primary" type="submit">Send</button></form>
        <div class="examples"><b>Ask naturally:</b> Who is here? What should I remember about him? Am I seeing anyone today? Who am I meeting with?<br><b>Book by voice:</b> Book a meeting with Alex tomorrow at 3 PM.<br><b>Save person:</b> Save this person as Alex, he is my friend.<br><b>Reminder:</b> Remember that Alex is bringing the documents tomorrow.<br><b>Edit:</b> Edit reminder to bring the documents tomorrow. Edit name to Alex. Edit relationship to friend.</div>
        <div class="foot">The PC must remain on. Camera frames, face templates, speech recognition, Ollama, and MongoDB are processed on the PC. To use the phone camera or mic, open this page over HTTPS. Use a private Tailscale connection; do not expose the app directly to the internet.</div>
      </div>
    </section>
  </main>
  <div id="editDialog" class="modal" hidden role="dialog" aria-modal="true" aria-labelledby="editTitle">
    <form id="editProfileForm" class="modal-card">
      <h2 id="editTitle">Edit recognized person</h2>
      <label>Name<input id="editName" maxlength="80" required></label>
      <label>Relationship<input id="editRelationship" maxlength="80" placeholder="For example: friend"></label>
      <label>Reminder<textarea id="editReminder" maxlength="500" rows="4" placeholder="For example: Bring the documents tomorrow"></textarea></label>
      <div class="modal-actions"><button class="button" type="button" onclick="closeEditDialog()">Cancel</button><button class="button primary" type="submit">Save changes</button></div>
    </form>
  </div>
  <div id="enrollmentDialog" class="modal" hidden role="dialog" aria-modal="true" aria-labelledby="enrollmentTitle">
    <form id="enrollmentForm" class="modal-card">
      <h2 id="enrollmentTitle">Confirm the new person's name</h2>
      <p class="sub">Speech recognition can mishear names. Check the spelling before saving this face.</p>
      <label>Name<input id="enrollmentName" maxlength="80" required></label>
      <label>Relationship (optional)<input id="enrollmentRelationship" maxlength="80" placeholder="For example: friend"></label>
      <div class="modal-actions"><button class="button" type="button" onclick="closeEnrollmentDialog()">Cancel</button><button class="button primary" type="submit">Confirm and save face</button></div>
    </form>
  </div>
  <div id="appointmentDialog" class="modal" hidden role="dialog" aria-modal="true" aria-labelledby="appointmentTitle">
    <form id="appointmentForm" class="modal-card">
      <h2 id="appointmentTitle">Book an appointment or meeting</h2>
      <label>What is it for?<input id="appointmentSummary" maxlength="160" placeholder="For example: Project discussion" required></label>
      <label>Who is it with?<input id="appointmentPerson" maxlength="80" placeholder="Person's name" required></label>
      <label>Date<input id="appointmentDate" type="date" required></label>
      <label>Time<input id="appointmentTime" type="time" required></label>
      <div class="appointment-list"><h3>Upcoming appointments</h3><ul id="appointmentList"><li>Loading appointments…</li></ul></div>
      <div class="modal-actions"><button class="button" type="button" onclick="closeAppointmentDialog()">Close</button><button class="button primary" type="submit">Save appointment</button></div>
    </form>
  </div>
<script>
const $=id=>document.getElementById(id);let lastSpeech=0,lastRunning=false,lastState=null,phoneStream=null,frameTimer=null,audioContext=null,audioProcessor=null,audioBusy=false,pcmPending=new Float32Array(0),speechVoices=[],switchingPhoneCamera=false;
function voiceScore(v){const n=(v.name+' '+v.voiceURI).toLowerCase(),lang=(v.lang||'').toLowerCase();let score=lang.startsWith('en')?10:0;if(/female|woman|samantha|jenny|aria|ava|allison|karen|moira|tessa|victoria|zira|sonia|libby|susan|hazel|serena|natasha|siri/.test(n))score+=30;if(/male|david|mark|guy|daniel|alex|george|james/.test(n))score-=25;if(v.localService)score+=2;return score}
function loadSpeechVoices(){if(!('speechSynthesis'in window))return;speechVoices=speechSynthesis.getVoices();const select=$('speechVoice'),saved=localStorage.getItem('visitorAssistantSpeechVoice')||'auto';select.innerHTML='<option value="auto">Auto (pleasant English voice)</option>';const english=speechVoices.filter(v=>(v.lang||'').toLowerCase().startsWith('en')).sort((a,b)=>voiceScore(b)-voiceScore(a)||a.name.localeCompare(b.name));for(const v of english){const o=document.createElement('option');o.value=v.voiceURI;o.textContent=`${v.name} (${v.lang})`;select.appendChild(o)}select.value=[...select.options].some(o=>o.value===saved)?saved:'auto'}
function say(text){if(!('speechSynthesis'in window)||!text)return;speechSynthesis.cancel();const utterance=new SpeechSynthesisUtterance(text),choice=$('speechVoice').value;let voice=choice==='auto'?speechVoices.filter(v=>(v.lang||'').toLowerCase().startsWith('en')).sort((a,b)=>voiceScore(b)-voiceScore(a))[0]:speechVoices.find(v=>v.voiceURI===choice);if(voice){utterance.voice=voice;utterance.lang=voice.lang}utterance.rate=.96;utterance.pitch=1.04;speechSynthesis.speak(utterance)}
if('speechSynthesis'in window){loadSpeechVoices();speechSynthesis.onvoiceschanged=loadSpeechVoices} $('speechVoice').addEventListener('change',e=>localStorage.setItem('visitorAssistantSpeechVoice',e.target.value));
async function api(path,body){let r=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});let data=await r.json().catch(()=>({}));if(!r.ok)throw Error(data.error||'Request failed');return data}
function openEditDialog(s=lastState){if(!s||!s.person||['Unrecognized visitor','Recent unrecognized visitor','No known visitor','Waiting for camera','Camera off'].includes(s.person))return;$('editName').value=s.person;$('editRelationship').value=s.relationship||'';$('editReminder').value=s.profile_reminder||'';$('editDialog').hidden=false;$('editName').focus()}
function closeEditDialog(){$('editDialog').hidden=true}
function closeEnrollmentDialog(){$('enrollmentDialog').hidden=true}
function closeAppointmentDialog(){$('appointmentDialog').hidden=true}
async function openAppointmentDialog(){const person=lastState?.person||'';$('appointmentPerson').value=['Unrecognized visitor','Recent unrecognized visitor','No known visitor','Waiting for camera','Camera off'].includes(person)?'':person;$('appointmentDate').value='';$('appointmentTime').value='';$('appointmentSummary').value='';$('appointmentDialog').hidden=false;await loadAppointments();$('appointmentSummary').focus()}
async function loadAppointments(){const list=$('appointmentList');try{const data=await api('/api/appointments');list.replaceChildren();if(!data.appointments.length){const li=document.createElement('li');li.textContent='No upcoming appointments saved.';list.appendChild(li);return}for(const item of data.appointments){const li=document.createElement('li'),title=document.createElement('strong'),detail=document.createElement('span');title.textContent=item.summary||'Appointment';detail.textContent=[item.with_person,item.date_text,item.time_text].filter(Boolean).join(' · ');li.append(title,detail);list.appendChild(li)}}catch(e){list.replaceChildren();const li=document.createElement('li');li.textContent=e.message;list.appendChild(li)}}
document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeEditDialog();closeEnrollmentDialog();closeAppointmentDialog()}});
document.addEventListener('click',e=>{if(e.target===$('editDialog'))closeEditDialog();if(e.target===$('enrollmentDialog'))closeEnrollmentDialog();if(e.target===$('appointmentDialog'))closeAppointmentDialog()});
async function saveEditedProfile(e){e.preventDefault();if(!lastState)return;try{const result=await api('/api/profile',{old_name:lastState.person,name:$('editName').value.trim(),relationship:$('editRelationship').value.trim(),reminder:$('editReminder').value.trim()});closeEditDialog();$('notice').textContent=result.message||'Person details updated.';await refresh()}catch(err){$('notice').textContent=err.message}}
$('editProfileForm').addEventListener('submit',saveEditedProfile);
async function saveEnrollment(e){e.preventDefault();const button=$('enrollmentForm').querySelector('button[type=submit]');button.disabled=true;try{const result=await api('/api/enroll',{name:$('enrollmentName').value.trim(),relationship:$('enrollmentRelationship').value.trim()});closeEnrollmentDialog();$('notice').textContent=result.message||'Person saved.';await refresh()}catch(err){$('notice').textContent=err.message}finally{button.disabled=false}}
$('enrollmentForm').addEventListener('submit',saveEnrollment);
async function saveAppointment(e){e.preventDefault();const button=$('appointmentForm').querySelector('button[type=submit]');button.disabled=true;try{const result=await api('/api/appointments',{summary:$('appointmentSummary').value.trim(),with_person:$('appointmentPerson').value.trim(),date_iso:$('appointmentDate').value,time_24h:$('appointmentTime').value});$('notice').textContent=result.message||'Appointment saved.';$('appointmentForm').reset();await loadAppointments();await refresh()}catch(err){$('notice').textContent=err.message}finally{button.disabled=false}}
$('appointmentForm').addEventListener('submit',saveAppointment);
async function control(action){try{await api('/api/control',{action});if(action==='stop')stopPhone();await refresh()}catch(e){$('notice').textContent=e.message}}
async function startPhone(){if(!window.isSecureContext||!navigator.mediaDevices){$('notice').textContent='Phone camera and mic need a secure HTTPS address. Use the Tailscale HTTPS link, not the local http address.';return}try{const facing=$('phoneFacing').value;phoneStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facing},width:{ideal:640},height:{ideal:480}},audio:{echoCancellation:true,noiseSuppression:true}});$('phoneCamera').srcObject=phoneStream;await $('phoneCamera').play();await api('/api/control',{action:'start_phone'});startPhoneFrameUpload();startPhoneAudioUpload();$('notice').textContent=`Phone ${facing==='user'?'front':'rear'} camera and microphone connected. Keep one unfamiliar face in view.`;await refresh()}catch(e){stopPhone();$('notice').textContent='Could not start phone camera/mic: '+e.message}}
async function switchPhoneCamera(){if(!phoneStream||switchingPhoneCamera)return;switchingPhoneCamera=true;$('switchPhone').disabled=true;try{await api('/api/control',{action:'stop'});stopPhone();$('phoneFacing').value=$('phoneFacing').value==='environment'?'user':'environment';await startPhone()}catch(e){$('notice').textContent='Could not switch phone camera: '+e.message}finally{switchingPhoneCamera=false;refresh()}}
function stopPhone(){if(frameTimer){clearInterval(frameTimer);frameTimer=null}if(audioProcessor){audioProcessor.disconnect();audioProcessor=null}if(audioContext){audioContext.close().catch(()=>{});audioContext=null}if(phoneStream){phoneStream.getTracks().forEach(t=>t.stop());phoneStream=null}pcmPending=new Float32Array(0);$('phoneCamera').srcObject=null}
function startPhoneFrameUpload(){const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');let busy=false;frameTimer=setInterval(()=>{if(busy||!phoneStream)return;const v=$('phoneCamera');if(!v.videoWidth)return;busy=true;canvas.width=Math.min(v.videoWidth,640);canvas.height=Math.round(v.videoHeight*canvas.width/v.videoWidth);ctx.drawImage(v,0,0,canvas.width,canvas.height);canvas.toBlob(async blob=>{try{if(blob)await fetch('/api/phone-frame',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob})}catch{}finally{busy=false}},'image/jpeg',.72)},280)}
function downsample(input,fromRate,toRate){if(fromRate===toRate)return input;const ratio=fromRate/toRate,outLength=Math.floor(input.length/ratio),out=new Float32Array(outLength);for(let i=0;i<outLength;i++){const start=Math.floor(i*ratio),end=Math.floor((i+1)*ratio);let sum=0;for(let j=start;j<end;j++)sum+=input[j];out[i]=sum/Math.max(1,end-start)}return out}
function startPhoneAudioUpload(){audioContext=new AudioContext();audioContext.resume();const source=audioContext.createMediaStreamSource(phoneStream),muter=audioContext.createGain();muter.gain.value=0;audioProcessor=audioContext.createScriptProcessor(4096,1,1);audioProcessor.onaudioprocess=e=>{const chunk=downsample(e.inputBuffer.getChannelData(0),audioContext.sampleRate,16000),joined=new Float32Array(pcmPending.length+chunk.length);joined.set(pcmPending);joined.set(chunk,pcmPending.length);pcmPending=joined;if(pcmPending.length>=16000&&!audioBusy){const n=16000,pcm=new Int16Array(n);for(let i=0;i<n;i++)pcm[i]=Math.max(-1,Math.min(1,pcmPending[i]))*32767;pcmPending=pcmPending.slice(n);audioBusy=true;fetch('/api/phone-audio',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:pcm.buffer}).catch(()=>{}).finally(()=>audioBusy=false)}};source.connect(audioProcessor);audioProcessor.connect(muter);muter.connect(audioContext.destination)}
async function refresh(){try{let s=await api('/api/state'),previousState=lastState;lastState=s;$('cameraStatus').textContent=s.running?(s.paused?'Recognition paused':'Camera on'):'Camera off';$('cameraStatus').className='status '+(s.running&&!s.paused?'on':'');$('start').textContent=s.running?'Stop camera':'Start on PC';$('start').onclick=()=>control(s.running?'stop':'start');$('startPhone').textContent=s.running&&s.input_mode==='phone'?'Stop phone camera & mic':'Use phone camera & mic';$('startPhone').disabled=s.running&&s.input_mode!=='phone';$('startPhone').onclick=()=>s.running?control('stop'):startPhone();$('phoneFacing').disabled=s.running;$('switchPhone').disabled=switchingPhoneCamera||!(s.running&&s.input_mode==='phone');if(s.running&&!lastRunning)$('video').src='/video?t='+Date.now();lastRunning=s.running;$('pause').textContent=s.paused?'Resume recognition':'Pause recognition';$('mute').textContent=s.muted?'Unmute speech':'Mute speech';$('person').textContent=s.person||'Waiting for camera';$('relationship').textContent=s.relationship||'';$('editProfile').disabled=!s.person||['Unrecognized visitor','Recent unrecognized visitor','No known visitor','Waiting for camera','Camera off'].includes(s.person);$('count').textContent=`Face samples: ${s.samples}/${s.samples_required}`;$('notice').textContent=s.notice||'Ready';$('notice').className='notice '+(s.sample_ready?'ready':'');$('reminder').textContent=s.reminder||'No reminder yet.';$('voiceStatus').textContent=(s.voice_error?`Microphone unavailable: ${s.voice_error}`:(s.voice_ready?`${s.input_mode==='phone'?'Phone':'PC'} microphone listening locally.`:`${s.input_mode==='phone'?'Phone':'PC'} microphone is starting or unavailable.`))+(s.last_transcript?` Heard: “${s.last_transcript}”`:``);if(previousState&&s.edit_requested>previousState.edit_requested)openEditDialog(s);if(previousState&&s.enrollment_requested>previousState.enrollment_requested&&s.pending_enrollment){$('enrollmentName').value=s.pending_enrollment.name||'';$('enrollmentRelationship').value=s.pending_enrollment.relationship||'';$('enrollmentDialog').hidden=false;$('enrollmentName').focus()}if(s.speech_id>lastSpeech){lastSpeech=s.speech_id;if($('speech').checked)say(s.speech_text||'')}}catch(e){$('notice').textContent=e.message}}
$('commandForm').addEventListener('submit',async e=>{e.preventDefault();let text=$('command').value.trim();if(!text)return;try{let r=await api('/api/command',{text});$('command').value='';$('notice').textContent=r.message||'Done.';await refresh()}catch(err){$('notice').textContent=err.message}});
$('video').src='/video';setInterval(refresh,900);refresh();
</script>
</body></html>'''


LOGIN_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SPONGE access</title><style>
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0b1220;color:#eef4ff;font:16px Segoe UI,system-ui,sans-serif;padding:18px}.card{max-width:410px;width:100%;padding:25px;border:1px solid #263650;border-radius:18px;background:#111c2f}h1{margin:0 0 8px;font-size:24px}p{color:#a7b5ca;line-height:1.5}input,button{width:100%;padding:13px;border-radius:11px;font:16px Segoe UI,system-ui}input{background:#0a1424;color:white;border:1px solid #263650;margin:8px 0 10px}button{border:0;background:#4d7de0;color:white;font-weight:700}.error{color:#ffadad;min-height:22px}</style></head><body><form class="card" method="post" action="/login"><h1>SPONGE</h1><p>Enter the access code shown in the PC terminal. This keeps other devices on your Wi‑Fi from opening the camera and reminders.</p><div class="error">{{ERROR}}</div><input name="pin" type="password" inputmode="numeric" autocomplete="one-time-code" placeholder="Access code" required autofocus><button>Connect</button></form></body></html>'''


class AssistantService:
    def __init__(self):
        self.state = {
            "running": False, "stop": False, "paused": False, "muted": False,
            "speaking": False, "voice_ready": False, "voice_error": "",
            "current_visitor": "Waiting for camera", "relationship": "",
            "profile_reminder": "", "edit_requested": 0,
            "enrollment_requested": 0, "pending_enrollment": None,
            "pending_appointment": None,
            "last_reminder": "Start the camera to see visitor reminders here.",
            "candidate_samples": [], "candidate_embedding": None,
            "candidate_last_seen": 0.0, "candidate_sampling_notified": False,
            "candidate_ready_notified": False, "notice": "Start the camera on your PC to begin.",
            "speech_text": "", "speech_id": 0, "people_changed": False,
            "conversation_history": [],
            "last_transcript": "",
        }
        self.lock = threading.RLock()
        self.frame_condition = threading.Condition()
        self.latest_jpeg = None
        self.model_data = None
        self.phone_model = None
        self.camera_thread = None
        self.voice_worker = None
        self.phone_recognizer = None
        self.phone_recognizer_lock = threading.Lock()
        self.busy = set()
        self.announced_people = set()

    def set_notice(self, text, speak=False, ready=False):
        with self.lock:
            self.state["notice"] = text
            if speak and not self.state["muted"]:
                self.state["speech_text"] = text
                self.state["speech_id"] += 1
            if ready:
                self.state["candidate_ready_notified"] = True

    def publish_response(self, text, remember=False):
        with self.lock:
            if remember:
                self.state["last_reminder"] = text
            self.state["notice"] = text
            if not self.state["muted"]:
                self.state["speech_text"] = text
                self.state["speech_id"] += 1
            self.state["conversation_history"].append({"role": "assistant", "content": text})
            self.state["conversation_history"] = self.state["conversation_history"][-20:]

    def start(self, input_mode="pc"):
        with self.lock:
            if self.state["running"]:
                return
            self.announced_people.clear()
            self.state.update(running=True, stop=False, paused=False, voice_error="", voice_ready=False, input_mode=input_mode, phone_frame=None, phone_frame_at=0.0)
            self.state["notice"] = "Starting camera and local voice recognition…" if input_mode == "pc" else "Connecting to the phone camera and loading local voice recognition…"
        self.camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.camera_thread.start()

    def update_person_profile(self, old_name, name, relationship, reminder):
        old_name = str(old_name).strip()
        name = str(name).strip()
        relationship = str(relationship).strip()
        reminder = str(reminder).strip()
        if not old_name or not name or len(name) > 80:
            raise ValueError("Enter a valid person name.")
        if len(relationship) > 80 or len(reminder) > 500:
            raise ValueError("Relationship must be under 80 characters and reminder under 500 characters.")
        if old_name != name and name in load_people():
            raise ValueError(f"A person named {name} is already saved.")

        client, database = connect_database()
        try:
            existing = database.people.find_one({"name": name}) if name != old_name else None
            if existing:
                raise ValueError(f"A person named {name} is already saved.")
            database.people.update_one(
                {"name": old_name},
                {"$set": {"name": name, "relationship": relationship, "reminder": reminder}},
                upsert=True,
            )
            if old_name != name:
                database.memories.update_many({"name": old_name}, {"$set": {"name": name}})
                database.meetings.update_many({"with_person": old_name}, {"$set": {"with_person": name}})
        finally:
            client.close()

        people = load_people()
        embedding = people.pop(old_name, None)
        if embedding is not None:
            people[name] = embedding
            temporary_path = PEOPLE_PATH.with_suffix(".tmp")
            temporary_path.write_text(json.dumps(people, indent=2), encoding="utf-8")
            temporary_path.replace(PEOPLE_PATH)
        with self.lock:
            self.state["current_visitor"] = name
            self.state["current_person_for_question"] = name
            self.state["relationship"] = relationship
            self.state["profile_reminder"] = reminder
            self.state["people_changed"] = old_name != name
            if old_name != name:
                self.announced_people.discard(old_name)
                self.announced_people.add(name)
            self.state["last_reminder"] = self.arrival_text(name, relationship, reminder)
            message = f"Updated {name}'s details."
            self.state["notice"] = message
        return message

    def confirm_person_enrollment(self, name, relationship):
        name = str(name).strip()
        relationship = str(relationship).strip()
        if not name or len(name) > 80 or len(relationship) > 80:
            raise ValueError("Enter a valid name and a relationship under 80 characters.")
        with self.lock:
            pending = self.state.get("pending_enrollment")
            candidate = self.state.get("candidate_embedding")
        if not pending or candidate is None:
            raise ValueError("The face sample expired. Keep the person in view and say the save command again.")
        if name in load_people():
            raise ValueError(f"{name} is already saved. Use Edit profile to change that person's details.")
        save_profile(name, relationship)
        people = load_people()
        people[name] = candidate
        temporary_path = PEOPLE_PATH.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(people, indent=2), encoding="utf-8")
        temporary_path.replace(PEOPLE_PATH)
        with self.lock:
            self.state["candidate_samples"] = []
            self.state["candidate_embedding"] = None
            self.state["candidate_sampling_notified"] = False
            self.state["candidate_ready_notified"] = False
            self.state["pending_enrollment"] = None
            self.state["people_changed"] = True
            self.state["current_visitor"] = name
            self.state["current_person_for_question"] = name
            self.state["relationship"] = relationship
        relation_text = f", your {relationship}" if relationship else ""
        return f"Saved {name}{relation_text}. I'll recognize them next time."

    @staticmethod
    def arrival_text(name, relationship, reminder):
        arrival = f"{name}, your {relationship}, is here." if relationship else f"{name} is here."
        return f"{arrival} Reminder: {reminder}" if reminder else arrival

    def stop(self):
        with self.lock:
            self.state["stop"] = True
            self.state["notice"] = "Stopping camera…"

    def camera_worker(self):
        camera = None
        last_seen = {}
        people = {}
        last_people_check = 0.0
        try:
            if self.model_data is None:
                self.model_data = load_models()
            detector, recognizer = self.model_data
            people = load_people()
            with self.lock:
                input_mode = self.state["input_mode"]
            if input_mode == "pc":
                camera = open_camera()
                self.voice_worker = start_voice_transcriber(
                    self.state, lambda phrase: threading.Thread(
                        target=self.handle_command, args=(phrase,), daemon=True
                    ).start()
                )
                self.set_notice("Camera on. PC microphone is listening locally.")
            else:
                try:
                    from vosk import KaldiRecognizer, Model
                    if not VOICE_MODEL_PATH.exists():
                        raise RuntimeError(f"Local voice model not found: {VOICE_MODEL_PATH}")
                    with self.phone_recognizer_lock:
                        self.phone_model = Model(str(VOICE_MODEL_PATH))
                        self.phone_recognizer = KaldiRecognizer(self.phone_model, 16000)
                    with self.lock:
                        self.state["voice_ready"] = True
                        self.state["voice_error"] = ""
                    self.set_notice("Phone camera connected. Phone speech is being recognized locally on the PC.")
                except Exception as exc:
                    with self.lock:
                        self.state["voice_error"] = str(exc)
                    self.set_notice("Phone camera connected, but local phone-mic recognition could not start.")
            failed_reads = 0
            while not self.state["stop"]:
                if input_mode == "pc":
                    ok, frame = camera.read()
                    if not ok:
                        failed_reads += 1
                        if failed_reads >= 50:
                            raise RuntimeError("The webcam stopped sending video. Check camera permission and close other camera apps.")
                        time.sleep(0.02)
                        continue
                    failed_reads = 0
                else:
                    with self.lock:
                        frame_at = self.state.get("phone_frame_at", 0.0)
                        phone_frame = self.state.get("phone_frame")
                        if phone_frame is not None and time.monotonic() - frame_at > 3:
                            phone_frame = None
                    if phone_frame is None:
                        time.sleep(0.04)
                        continue
                    frame = phone_frame.copy()
                now = time.monotonic()
                if now - last_people_check > 0.5:
                    with self.lock:
                        changed = self.state.pop("people_changed", False)
                    if changed:
                        people = load_people()
                    last_people_check = now

                with self.lock:
                    paused = self.state["paused"]
                faces = None if paused else self._detect(detector, frame)
                names = []
                unknown_count = 0
                if faces is not None:
                    for face in faces:
                        x, y, width, height = [int(value) for value in face[:4]]
                        current = face_embedding(recognizer, frame, face)
                        best_name, best_score = "Unknown", -1.0
                        for name, saved_embedding in people.items():
                            saved = np.asarray(saved_embedding, dtype=np.float32).reshape(1, -1)
                            score = float(recognizer.match(current, saved, cv2.FaceRecognizerSF_FR_COSINE))
                            if score > best_score:
                                best_name, best_score = name, score
                        if best_score < MATCH_THRESHOLD:
                            best_name = "Unknown"
                            unknown_count += 1
                        else:
                            names.append(best_name)
                            if now - last_seen.get(best_name, 0.0) > 10:
                                last_seen[best_name] = now
                                self.request_reminder(best_name)
                        color = (56, 190, 120) if best_name != "Unknown" else (25, 165, 245)
                        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                        cv2.putText(frame, best_name, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

                self.collect_candidate(faces, frame, recognizer, unknown_count)
                with self.lock:
                    samples = self.state["candidate_samples"]
                    count = len(samples)
                    if not paused:
                        if names:
                            self.state["candidate_samples"] = []
                            self.state["candidate_embedding"] = None
                            self.state["candidate_sampling_notified"] = False
                            self.state["candidate_ready_notified"] = False
                            self.state["current_visitor"] = names[0] if len(names) == 1 else f"{len(names)} recognized visitors"
                            current_name = names[0]
                        elif unknown_count:
                            if faces is not None and len(faces) > 1:
                                self.state["candidate_samples"] = []
                                self.state["candidate_embedding"] = None
                                self.state["candidate_sampling_notified"] = False
                                self.state["candidate_ready_notified"] = False
                            self.state["current_visitor"] = "Unrecognized visitor"
                            current_name = ""
                        else:
                            age = now - self.state.get("candidate_last_seen", 0.0)
                            current_name = ""
                            if not self.state["candidate_samples"] or age > 60:
                                self.state["candidate_samples"] = []
                                self.state["candidate_embedding"] = None
                                self.state["candidate_sampling_notified"] = False
                                self.state["candidate_ready_notified"] = False
                                self.state["current_visitor"] = "No known visitor"
                            else:
                                self.state["current_visitor"] = "Recent unrecognized visitor"
                        if names:
                            self.state["candidate_samples"] = []
                            self.state["candidate_embedding"] = None
                        else:
                            self.state["profile_reminder"] = ""
                        self.state["relationship"] = "" if names else f"{count}/{SAMPLES_TO_COLLECT} temporary samples · not saved"
                        self.state["current_person_for_question"] = current_name or (names[0] if names and len(names) == 1 else "")
                    if not self.state["paused"]:
                        cv2.putText(frame, f"Face samples: {count}/{SAMPLES_TO_COLLECT}", (14, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 240, 255), 2, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    with self.frame_condition:
                        self.latest_jpeg = encoded.tobytes()
                        self.frame_condition.notify_all()
                time.sleep(0.015)
        except Exception as exc:
            self.set_notice(str(exc))
        except SystemExit as exc:
            self.set_notice(str(exc))
        finally:
            with self.lock:
                self.state["stop"] = True
            if self.voice_worker is not None and self.voice_worker.is_alive():
                self.voice_worker.join(timeout=1)
            if camera is not None:
                camera.release()
            with self.lock:
                self.state["running"] = False
                self.state["voice_ready"] = False
                self.state["current_visitor"] = "Camera off"
                self.state["phone_frame"] = None
            with self.frame_condition:
                self.latest_jpeg = None
                self.frame_condition.notify_all()

    @staticmethod
    def _detect(detector, frame):
        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        return detector.detect(frame)[1]

    def collect_candidate(self, faces, frame, recognizer, unknown_count):
        if faces is None or len(faces) != 1 or unknown_count != 1:
            return
        with self.lock:
            samples = self.state["candidate_samples"]
            now = time.monotonic()
            if samples and now - self.state.get("candidate_last_seen", 0.0) > 3:
                samples.clear()
                self.state["candidate_embedding"] = None
                self.state["candidate_sampling_notified"] = False
                self.state["candidate_ready_notified"] = False
            if now < self.state.get("next_candidate_sample", 0):
                return
        sample = face_embedding(recognizer, frame, faces[0])
        with self.lock:
            samples = self.state["candidate_samples"]
            if samples:
                sample_vector = sample.reshape(-1)
                prior = np.mean(np.stack(samples), axis=0).reshape(-1)
                similarity = float(np.dot(sample_vector, prior) / max(float(np.linalg.norm(sample_vector) * np.linalg.norm(prior)), 1e-12))
                if similarity < MATCH_THRESHOLD:
                    samples.clear()
                    self.state["candidate_embedding"] = None
                    self.state["candidate_sampling_notified"] = False
                    self.state["candidate_ready_notified"] = False
            samples.append(sample)
            if len(samples) > SAMPLES_TO_COLLECT:
                del samples[0]
            self.state["next_candidate_sample"] = now + 0.3
            self.state["candidate_last_seen"] = now
            first_sample = not self.state.get("candidate_sampling_notified")
            if first_sample:
                self.state["candidate_sampling_notified"] = True
            if len(samples) >= SAMPLES_TO_COLLECT:
                average = np.mean(np.stack(samples), axis=0)
                average /= max(float(np.linalg.norm(average)), 1e-12)
                self.state["candidate_embedding"] = average.reshape(-1).tolist()
                ready = not self.state.get("candidate_ready_notified")
                self.state["candidate_ready_notified"] = True
            else:
                ready = False
            count = len(samples)
        if first_sample:
            self.set_notice(f"Unfamiliar face detected. Sampling started ({count}/{SAMPLES_TO_COLLECT}); keep one face in view.", speak=True)
        if ready:
            self.set_notice(f"Face sample ready ({SAMPLES_TO_COLLECT}/{SAMPLES_TO_COLLECT}). You can save now: say ‘Save this person as Alex, he is my friend.’", speak=True, ready=True)

    def request_reminder(self, name):
        with self.lock:
            if name in self.busy or name in self.announced_people:
                return
            self.busy.add(name)
            self.announced_people.add(name)
            self.state["paused"] = True
            self.state["notice"] = f"Recognized {name}. Recognition paused. Ask about this person or say ‘repeat’ to hear the reminder."

        def work():
            try:
                message = generate_notification(name)
                client, database = connect_database()
                try:
                    profile = database.people.find_one({"name": name}) or {}
                finally:
                    client.close()
                relationship = profile.get("relationship", "")
                profile_reminder = profile.get("reminder", "")
            except Exception as exc:
                message = f"{name} is here."
                relationship = ""
                profile_reminder = ""
                self.set_notice(f"Using a simple arrival message ({exc})")
            with self.lock:
                self.state["last_reminder"] = message
                self.state["current_visitor"] = name
                self.state["relationship"] = relationship
                self.state["profile_reminder"] = profile_reminder
                self.state["notice"] = message
                self.busy.discard(name)

        threading.Thread(target=work, daemon=True).start()

    def handle_command(self, text):
        phrase = " ".join(text.lower().strip().split()).strip(" .,!?;:")
        with self.lock:
            self.state["last_transcript"] = text.strip()
            self.state["conversation_history"].append({"role": "user", "content": text.strip()})
            self.state["conversation_history"] = self.state["conversation_history"][-20:]
        edit_text = re.sub(
            r"^(?:(?:hey|okay|ok)\s+assistant[, ]+)?(?:please\s+)?(?:(?:i\s+)?(?:want|need)\s+to\s+)?",
            "", text.strip(), flags=re.IGNORECASE,
        )
        edit_match = re.match(
            r"^(?:edit|change|update)(?:\s+(?:the|his|her|their))?\s+(reminder|name|relationship)(?:\s+(?:to|as|that says|to say))\s+(.+?)\s*[.!?]*$",
            edit_text, re.IGNORECASE,
        )
        open_edit_match = re.match(
            r"^(?:edit|change|update)\s+(?:(?:the|this)\s+)?(?:profile|person|reminder|name|relationship)$",
            edit_text, re.IGNORECASE,
        )
        if phrase in {"edit profile", "edit person", "edit this person", "edit reminder", "edit name", "edit relationship", "open edit dialog"} or open_edit_match:
            with self.lock:
                self.state["edit_requested"] += 1
            message = "Opening the edit box for the recognized person."
            self.publish_response(message, remember=False)
            return message
        if edit_match:
            field, value = edit_match.group(1).lower(), edit_match.group(2).strip(" .,!?:;")
            with self.lock:
                current_name = self.state.get("current_person_for_question") or self.state.get("current_visitor", "")
                relationship = self.state.get("relationship", "")
                reminder = self.state.get("profile_reminder", "")
            if current_name in {"Unrecognized visitor", "Recent unrecognized visitor", "No known visitor", "Waiting for camera", "Camera off", ""}:
                message = "I don't have a recognized person to edit yet."
            else:
                new_name = value if field == "name" else current_name
                new_relationship = value if field == "relationship" else relationship
                new_reminder = value if field == "reminder" else reminder
                try:
                    message = self.update_person_profile(current_name, new_name, new_relationship, new_reminder)
                except Exception as exc:
                    message = f"I couldn't update that: {exc}"
            self.publish_response(message, remember=False)
            return message
        controls = {
            "pause": "pause", "pause recognition": "pause",
            "resume": "resume", "resume recognition": "resume",
            "mute": "mute", "mute speech": "mute",
            "unmute": "unmute", "unmute speech": "unmute",
            "repeat": "repeat", "repeat it": "repeat", "repeat reminder": "repeat", "repeat that": "repeat",
            "say that again": "repeat", "repeat the reminder": "repeat", "can you repeat": "repeat",
            "can you repeat that": "repeat", "please repeat": "repeat",
            "stop": "stop", "stop assistant": "stop",
        }
        if phrase in controls:
            action = controls[phrase]
            with self.lock:
                if action == "pause":
                    self.state["paused"] = True
                    response = "Recognition paused."
                elif action == "resume":
                    self.state["paused"] = False
                    response = "Recognition resumed."
                elif action == "mute":
                    self.state["muted"] = True
                    response = "Speech muted."
                elif action == "unmute":
                    self.state["muted"] = False
                    response = "Speech unmuted."
                elif action == "repeat":
                    response = self.state.get("last_reminder", "There is no saved reminder yet.")
                else:
                    response = "Stopping the assistant."
            if action == "stop":
                self.stop()
            else:
                self.publish_response(response, remember=False)
            return response

        # Keep booking as a short spoken conversation when the first request
        # omits the person, date, or time instead of abandoning the booking.
        with self.lock:
            pending_appointment = dict(self.state.get("pending_appointment") or {})
        if pending_appointment and phrase in {"cancel", "cancel appointment", "cancel meeting", "never mind"}:
            with self.lock:
                self.state["pending_appointment"] = None
            message = "Okay, I cancelled that booking."
            self.publish_response(message, remember=False)
            return message
        if pending_appointment:
            known_names = list(load_people().keys())
            parsed = _extract_meeting_fields(text, known_names)
            for field in ("date_iso", "time_24h", "date_text", "time_text"):
                if parsed.get(field):
                    pending_appointment[field] = parsed[field]
            if not pending_appointment.get("meeting_with"):
                person_match = re.search(r"\b(?:with|meet|see)\s+(.+)$", text, re.IGNORECASE)
                guessed_person = next(
                    (name for name in known_names if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE)),
                    "",
                )
                if not guessed_person and person_match:
                    guessed_person = re.split(
                        r"\b(?:today|tomorrow|next|on|at|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                        person_match.group(1), maxsplit=1, flags=re.IGNORECASE,
                    )[0].strip(" .,!?;:")
                if not guessed_person and len(text.split()) <= 6:
                    remainder = re.sub(
                        r"\b(?:today|tomorrow|next|on|at|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                        r"january|february|march|april|may|june|july|august|september|october|november|december|"
                        r"a\.?m\.?|p\.?m\.?|noon|midnight|meeting|appointment)\b|\b\d{1,4}(?::\d{2})?\b",
                        "", text, flags=re.IGNORECASE,
                    ).strip(" .,!?;:")
                    if remainder and not parsed.get("date_iso") and not parsed.get("time_24h"):
                        guessed_person = remainder
                    elif remainder and (parsed.get("date_iso") or parsed.get("time_24h")):
                        guessed_person = remainder
                if guessed_person:
                    pending_appointment["meeting_with"] = guessed_person
            missing = [
                label for key, label in (("meeting_with", "who the appointment is with"),
                                         ("date_iso", "the date"))
                if not pending_appointment.get(key)
            ]
            if len(missing) < 2:
                with self.lock:
                    self.state["pending_appointment"] = pending_appointment
                if not missing:
                    save_appointment(
                        pending_appointment.get("summary", "Meeting"),
                        pending_appointment["meeting_with"],
                        pending_appointment["date_iso"],
                        pending_appointment["time_24h"],
                    )
                    message = (
                        f"Booked {pending_appointment.get('summary', 'Meeting')} with "
                        f"{pending_appointment['meeting_with']} for "
                        f"{pending_appointment.get('date_text') or pending_appointment['date_iso']}"
                        + (f" at {pending_appointment.get('time_text') or pending_appointment['time_24h']}" if pending_appointment.get("time_24h") else "; no time was specified")
                        + "."
                    )
                    with self.lock:
                        self.state["pending_appointment"] = None
                    self.publish_response(message, remember=False)
                    return message
                message = "Thanks. Please tell me " + ", then ".join(missing) + "."
                self.publish_response(message, remember=False)
                return message

        with self.lock:
            known_names = list(load_people().keys())
            current_person = self.state.get("current_visitor", "")
            intent_history = list(self.state["conversation_history"][:-1])
            if current_person in {"Unrecognized visitor", "Recent unrecognized visitor", "No known visitor", "Waiting for camera", "Camera off"}:
                current_person = ""
        request = interpret_voice_utterance(text, known_names, current_person, intent_history)
        intent = request.get("intent")
        if intent in {"ignore", "conversation"}:
            with self.lock:
                history = list(self.state["conversation_history"][:-1])
                context_name = self.state.get("current_person_for_question") or self.state.get("current_visitor", "")
                relationship = self.state.get("relationship", "")
                reminder = self.state.get("profile_reminder", "")
            if context_name in {"Unrecognized visitor", "Recent unrecognized visitor", "No known visitor", "Waiting for camera", "Camera off"}:
                context_name = ""
            message = answer_conversationally(text, context_name, relationship, reminder, history)
            self.publish_response(message, remember=False)
            return message
        if intent == "save_person":
            name = request.get("name", "").strip()
            relationship = request.get("relationship", "").strip()
            with self.lock:
                candidate = self.state.get("candidate_embedding")
            if not name:
                message = "I didn't catch the person's name. Please repeat it."
            elif not candidate:
                message = "No face sample is ready yet. Keep one unfamiliar face in view until you hear the ready message, then try again."
            else:
                # Speech recognition often mishears Indian names. Let the user
                # correct the transcript in the confirmation dialog before we
                # commit the face template to the saved profiles.
                with self.lock:
                    self.state["pending_enrollment"] = {
                        "name": name, "relationship": relationship,
                    }
                    self.state["enrollment_requested"] += 1
                message = f"I heard {name}. Please confirm or correct the name in the save dialog."
        elif intent == "remember":
            memory = request.get("memory", "").strip()
            if not memory:
                message = "I didn't catch the reminder. Please repeat it, starting with ‘Remember that…’"
            else:
                name = request.get("name", "").strip() or current_person or "General"
                appointment_request = request.get("is_appointment") or any(
                    word in text.lower() for word in ("meeting", "appointment")
                )
                if appointment_request:
                    appointment_summary = "Appointment" if "appointment" in text.lower() else "Meeting"
                    with_person = request.get("meeting_with", "").strip()
                    date_iso = request.get("date_iso", "").strip()
                    time_24h = request.get("time_24h", "").strip()
                    missing = []
                    if not with_person:
                        missing.append("who it is with")
                    if not date_iso:
                        missing.append("the date")
                    if missing:
                        with self.lock:
                            self.state["pending_appointment"] = {
                                "summary": appointment_summary,
                                "meeting_with": with_person,
                                "date_iso": date_iso,
                                "time_24h": time_24h,
                                "date_text": request.get("date_text", "").strip(),
                                "time_text": request.get("time_text", "").strip(),
                            }
                        message = "I can book that. Please tell me " + ", ".join(missing) + "."
                        self.publish_response(message, remember=False)
                        return message
                    save_appointment(appointment_summary, with_person, date_iso, time_24h)
                    date_label = request.get("date_text", "").strip() or date_iso
                    time_label = request.get("time_text", "").strip() or time_24h
                    time_details = f" at {time_label}" if time_label else "; no time was specified"
                    message = f"Booked your {appointment_summary.lower()} with {with_person} for {date_label}{time_details}."
                    self.publish_response(message, remember=False)
                    return message
                else:
                    save_memory(name, memory)
                    message = f"Got it. I’ll remember that for {name}."
                with self.lock:
                    self.state["last_reminder"] = memory
                    if name == self.state.get("current_person_for_question"):
                        self.state["profile_reminder"] = memory
                self.publish_response(message, remember=False)
                return message
        elif intent == "ask_meetings":
            message = answer_meeting_question(
                text, request.get("date_scope", ""), request.get("meeting_with", "")
            )
        elif intent == "ask_person":
            name = request.get("name", "").strip() or current_person
            with self.lock:
                history = list(self.state["conversation_history"][:-1])
                relationship = self.state.get("relationship", "") if name == current_person else ""
                reminder = self.state.get("profile_reminder", "") if name == current_person else ""
            message = answer_conversationally(text, name, relationship, reminder, history)
        else:
            message = "I didn't understand that request."
        self.publish_response(message, remember=False)
        return message

    def accept_phone_audio(self, audio_bytes):
        with self.lock:
            if not self.state["running"] or self.state.get("input_mode") != "phone":
                return
        with self.phone_recognizer_lock:
            recognizer = self.phone_recognizer
            if recognizer is None:
                return
            if recognizer.AcceptWaveform(audio_bytes):
                import json as json_module
                phrase = json_module.loads(recognizer.Result()).get("text", "").strip()
            else:
                phrase = ""
        if phrase:
            threading.Thread(target=self.handle_command, args=(phrase,), daemon=True).start()

    def snapshot(self):
        with self.lock:
            return {
                "running": self.state["running"], "paused": self.state["paused"],
                "muted": self.state["muted"], "voice_ready": self.state["voice_ready"],
                "input_mode": self.state.get("input_mode", "pc"),
                "voice_error": self.state["voice_error"], "person": self.state["current_visitor"],
                "relationship": self.state["relationship"], "reminder": self.state["last_reminder"],
                "samples": len(self.state["candidate_samples"]), "samples_required": SAMPLES_TO_COLLECT,
                "sample_ready": self.state["candidate_embedding"] is not None,
                "notice": self.state["notice"], "speech_text": self.state["speech_text"],
                "speech_id": self.state["speech_id"],
                "profile_reminder": self.state["profile_reminder"],
                "edit_requested": self.state["edit_requested"],
                "enrollment_requested": self.state["enrollment_requested"],
                "pending_enrollment": self.state.get("pending_enrollment"),
                "last_transcript": self.state["last_transcript"],
            }


service = AssistantService()


class WebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.access_pin = f"{secrets.randbelow(1_000_000):06d}"
        self.sessions = set()
        self.auth_lock = threading.Lock()
        self.failures = {}


class Handler(BaseHTTPRequestHandler):
    server_version = "VisitorAssistant/1.0"

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def session_id(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            return cookie.get("visitor_session").value if cookie.get("visitor_session") else ""
        except Exception:
            return ""

    def authorized(self):
        with self.server.auth_lock:
            return self.session_id() in self.server.sessions

    def read_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request is too large.")
        return self.rfile.read(length)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            if self.authorized():
                body = PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                body = LOGIN_PAGE.replace("{{ERROR}}", "").encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        if path == "/api/state":
            if not self.authorized():
                self.send_json({"error": "Enter the access code first."}, HTTPStatus.UNAUTHORIZED)
                return
            self.send_json(service.snapshot())
            return
        if path == "/api/appointments":
            if not self.authorized():
                self.send_json({"error": "Enter the access code first."}, HTTPStatus.UNAUTHORIZED)
                return
            try:
                self.send_json({"appointments": list_appointments()})
            except Exception as exc:
                self.send_json({"error": f"Could not load appointments: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/video":
            if not self.authorized():
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with service.frame_condition:
                        service.frame_condition.wait_for(lambda: service.latest_jpeg is not None or not service.state["running"], timeout=2)
                        frame = service.latest_jpeg
                        running = service.state["running"]
                    if frame is None:
                        if not running:
                            return
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
                    if not running:
                        return
                    time.sleep(0.06)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/login":
            try:
                form = parse_qs(self.read_body().decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            client_ip = self.client_address[0]
            now = time.monotonic()
            attempts = [value for value in self.server.failures.get(client_ip, []) if now - value < 60]
            self.server.failures[client_ip] = attempts
            if len(attempts) >= 6:
                body = LOGIN_PAGE.replace("{{ERROR}}", "Too many attempts. Wait one minute.").encode("utf-8")
                self.send_response(HTTPStatus.TOO_MANY_REQUESTS)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            pin = form.get("pin", [""])[0].strip()
            if not secrets.compare_digest(pin, self.server.access_pin):
                attempts.append(now)
                self.server.failures[client_ip] = attempts
                body = LOGIN_PAGE.replace("{{ERROR}}", "That code was not correct. Try again.").encode("utf-8")
                self.send_response(HTTPStatus.UNAUTHORIZED)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            session = secrets.token_urlsafe(32)
            with self.server.auth_lock:
                self.server.sessions.add(session)
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/")
            forwarded_proto = self.headers.get("X-Forwarded-Proto", "").casefold()
            secure_cookie = "; Secure" if forwarded_proto == "https" or self.headers.get("Host", "").casefold().endswith(".ts.net") else ""
            self.send_header("Set-Cookie", f"visitor_session={session}; HttpOnly; SameSite=Strict{secure_cookie}; Path=/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if not self.authorized():
            self.send_json({"error": "Enter the access code first."}, HTTPStatus.UNAUTHORIZED)
            return
        if path == "/api/phone-frame":
            try:
                raw = self.read_body()
                encoded = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("The phone sent an unreadable camera frame.")
                with service.lock:
                    if not service.state["running"] or service.state.get("input_mode") != "phone":
                        self.send_json({"error": "Start phone camera mode first."}, HTTPStatus.CONFLICT)
                        return
                    service.state["phone_frame"] = frame
                    service.state["phone_frame_at"] = time.monotonic()
                self.send_json({"ok": True})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/phone-audio":
            try:
                audio_bytes = self.read_body()
                service.accept_phone_audio(audio_bytes)
                self.send_json({"ok": True})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        try:
            data = json.loads(self.read_body().decode("utf-8"))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "Invalid request."}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/appointments":
            summary = str(data.get("summary", "")).strip()
            with_person = str(data.get("with_person", "")).strip()
            date_iso = str(data.get("date_iso", "")).strip()
            time_24h = str(data.get("time_24h", "")).strip()
            if len(summary) > 160 or len(with_person) > 80:
                self.send_json({"error": "Keep the appointment description under 160 characters and the name under 80."}, HTTPStatus.BAD_REQUEST)
                return
            try:
                save_appointment(summary, with_person, date_iso, time_24h)
                day = datetime.strptime(date_iso, "%Y-%m-%d").date()
                time_label = datetime.strptime(time_24h, "%H:%M").strftime("%I:%M %p").lstrip("0")
                date_label = day.strftime("%A, %B ") + str(day.day)
                message = f"Booked {summary} with {with_person} for {date_label} at {time_label}."
                service.publish_response(message, remember=False)
                self.send_json({"message": message})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.send_json({"error": f"Could not save appointment: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/control":
            action = data.get("action", "")
            if action == "start":
                service.start("pc")
                response = "Starting camera and local voice recognition."
            elif action == "start_phone":
                service.start("phone")
                response = "Starting phone camera and local microphone recognition."
            elif action == "stop":
                service.stop()
                response = "Stopping camera."
            elif action == "pause":
                with service.lock:
                    service.state["paused"] = not service.state["paused"]
                    paused = service.state["paused"]
                response = "Recognition paused." if paused else "Recognition resumed."
                service.publish_response(response, remember=False)
            elif action == "mute":
                with service.lock:
                    service.state["muted"] = not service.state["muted"]
                    muted = service.state["muted"]
                response = "Speech muted." if muted else "Speech unmuted."
                service.publish_response(response, remember=False)
            else:
                self.send_json({"error": "Unknown control."}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"message": response})
            return
        if path == "/api/command":
            text = str(data.get("text", "")).strip()
            if not text or len(text) > 500:
                self.send_json({"error": "Enter a command up to 500 characters."}, HTTPStatus.BAD_REQUEST)
                return
            try:
                response = service.handle_command(text)
                self.send_json({"message": response})
            except Exception as exc:
                response = f"I couldn't complete that request: {exc}"
                service.publish_response(response)
                self.send_json({"message": response}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/profile":
            old_name = str(data.get("old_name", "")).strip()
            name = str(data.get("name", "")).strip()
            relationship = str(data.get("relationship", "")).strip()
            reminder = str(data.get("reminder", "")).strip()
            try:
                response = service.update_person_profile(old_name, name, relationship, reminder)
                service.publish_response(response, remember=False)
                self.send_json({"message": response})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.send_json({"error": f"Could not save person details: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/enroll":
            name = str(data.get("name", "")).strip()
            relationship = str(data.get("relationship", "")).strip()
            try:
                message = service.confirm_person_enrollment(name, relationship)
                service.publish_response(message, remember=False)
                self.send_json({"message": message})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.send_json({"error": f"Could not save person: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def local_ip():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def main():
    server = WebServer(("0.0.0.0", PORT), Handler)
    print("SPONGE is ready.")
    print(f"On this PC: http://127.0.0.1:{PORT}")
    print(f"On the same Wi-Fi: http://{local_ip()}:{PORT}")
    print(f"Access code: {server.access_pin}")
    print("Keep this window open. Press Ctrl+C to stop the web app.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping SPONGE…")
    finally:
        service.stop()
        server.server_close()


if __name__ == "__main__":
    main()

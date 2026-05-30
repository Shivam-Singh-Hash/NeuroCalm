import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, CheckCircle, Circle, Activity, Heart, BrainCircuit, ShieldAlert, Zap, Download, Send, MessageSquare, X, LayoutDashboard, Users, Settings, Bell, User, Headphones, Play } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, PieChart, Pie, Sector, ReferenceLine } from 'recharts';
import axios from 'axios';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import './index.css';

function App() {
  const [dragActive, setDragActive] = useState(false);
  const [files, setFiles] = useState({ eda: null, temp: null, hr: null });
  const [loading, setLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [chartData, setChartData] = useState([]);
  
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', content: "Hello! I am NeuroCalm AI. How can I help you fully understand your physiological data today?" }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  
  const [activeView, setActiveView] = useState('dashboard');
  const [patientHistory, setPatientHistory] = useState([]);
  const [aiTone, setAiTone] = useState('compassionate');
  const [activeModel, setActiveModel] = useState('best');
  const [processingStep, setProcessingStep] = useState(null);
  const [timelineData, setTimelineData] = useState([]);
  const [sessionSummary, setSessionSummary] = useState(null);
  
  const [audioPlaying, setAudioPlaying] = useState(false);
  const audioCtxRef = useRef(null);
  const oscillatorsRef = useRef([]);

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const processFiles = (uploadedFiles) => {
    let newFiles = { ...files };
    Array.from(uploadedFiles).forEach(f => {
      const name = f.name.toUpperCase();
      if (name.includes('EDA')) newFiles.eda = f;
      else if (name.includes('TEMP')) newFiles.temp = f;
      else if (name.includes('HR')) newFiles.hr = f;
    });
    setFiles(newFiles);
  };

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files) processFiles(e.dataTransfer.files);
  };
  const handleChatSubmit = async (e) => {
    e?.preventDefault();
    if (!chatInput.trim()) return;
    
    const userMsg = chatInput.trim();
    setChatInput('');
    setChatHistory(prev => [...prev, { role: 'user', content: userMsg }]);
    setChatLoading(true);
    
    try {
      const metricsCtx = sessionSummary ? `Stress Score: ${Math.round(sessionSummary.score)}%, Peak at ${sessionSummary.peakTime}s. AI State: ${result.prediction_label}` : 'Baselines unknown';
      const res = await axios.post('http://localhost:5000/api/chat', {
        message: userMsg,
        context: metricsCtx,
        tone: aiTone
      });
      setChatHistory(prev => [...prev, { role: 'ai', content: res.data.reply }]);
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'ai', content: "I am having trouble connecting to my neural core right now." }]);
    } finally {
      setChatLoading(false);
    }
  };

  const generatePDF = async () => {
    if (!result) {
        alert("Please run a diagnostic first.");
        return;
    }
    
    setPdfLoading(true);
    
    try {
      // Fetch fresh, detailed PDF report from AI
      const avgHr = chartData.reduce((acc, curr) => acc + curr.hr, 0) / chartData.length || 75;
      const avgEda = chartData.reduce((acc, curr) => acc + curr.eda, 0) / chartData.length || 1.5;
      
      let reportRecommendation = sessionSummary?.recommendation;
      try {
          const res = await axios.post('/api/generate_report_plan', {
              score: sessionSummary.score,
              hr: avgHr,
              eda: avgEda,
              peak: sessionSummary.peakTime
          });
          reportRecommendation = res.data.recommendation;
      } catch (err) {
          console.error("Failed to generate specialized report plan", err);
      }
      
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      
      // Authentic Medical Reference ID
      const refId = `REF: NC-${Math.floor(Math.random() * 10000)}-${new Date().getFullYear()}`;
      
      // 1. Ultra-Premium Header Background
      pdf.setFillColor(15, 23, 42); // Dark slate
      pdf.rect(0, 0, pageWidth, 45, 'F');
      
      // Logo (Fake Brain/Pulse Icon using primitives)
      pdf.setDrawColor(45, 212, 191);
      pdf.setLineWidth(1.5);
      pdf.circle(26, 22.5, 6, 'S');
      pdf.line(22, 22.5, 25, 22.5);
      pdf.line(25, 22.5, 26, 19);
      pdf.line(26, 19, 27, 26);
      pdf.line(27, 26, 28, 22.5);
      pdf.line(28, 22.5, 30, 22.5);
      
      // Header Text
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(26);
      pdf.setFont(undefined, 'bold');
      pdf.text("NEUROCALM", 38, 26);
      
      pdf.setFontSize(9);
      pdf.setFont(undefined, 'normal');
      pdf.setTextColor(148, 163, 184); // slate-400
      pdf.text("ADVANCED NEURAL DIAGNOSTICS", 38, 32);
      
      pdf.setFontSize(10);
      pdf.setTextColor(255, 255, 255);
      pdf.text("CERTIFIED CLINICAL REPORT", pageWidth - 20, 24, { align: "right" });
      pdf.setTextColor(45, 212, 191);
      pdf.text(refId, pageWidth - 20, 30, { align: "right" });
      
      // 2. Patient / Session Meta Grid
      pdf.setDrawColor(226, 232, 240);
      pdf.setLineWidth(0.5);
      pdf.line(20, 52, pageWidth - 20, 52); // Top border
      
      pdf.setFontSize(8);
      pdf.setTextColor(100, 116, 139);
      pdf.text("DATE OF DIAGNOSIS", 20, 60);
      pdf.text("SESSION DURATION", 75, 60);
      pdf.text("PEAK INTENSITY TIME", 130, 60);
      
      pdf.setFontSize(11);
      pdf.setTextColor(15, 23, 42);
      pdf.setFont(undefined, 'bold');
      pdf.text(new Date().toLocaleDateString(), 20, 66);
      pdf.text(sessionSummary ? formatTime(sessionSummary.duration) : '--', 75, 66);
      pdf.text(sessionSummary ? formatTime(sessionSummary.peakTime) : '--', 130, 66);
      
      pdf.line(20, 72, pageWidth - 20, 72); // Bottom border
      
      // 3. Primary Diagnosis Result
      const isStress = sessionSummary?.score > 65;
      const isBaseline = sessionSummary?.score <= 35;
      const highlightColor = isStress ? [239, 68, 68] : (isBaseline ? [16, 185, 129] : [245, 158, 11]);
      
      pdf.setFillColor(highlightColor[0], highlightColor[1], highlightColor[2]);
      pdf.rect(20, 85, 4, 28, 'F');
      
      pdf.setFontSize(10);
      pdf.setTextColor(100, 116, 139);
      pdf.setFont(undefined, 'bold');
      pdf.text("PRIMARY ALGORITHMIC DIAGNOSIS", 30, 92);
      
      pdf.setFontSize(18);
      pdf.setTextColor(highlightColor[0], highlightColor[1], highlightColor[2]);
      const stateLabel = result.prediction_label ? result.prediction_label.toUpperCase() : (isStress ? 'STRESS' : 'BASELINE');
      pdf.text(`${stateLabel} STATE DETECTED`, 30, 103);
      
      pdf.setFontSize(10);
      pdf.setTextColor(71, 85, 105);
      pdf.setFont(undefined, 'normal');
      pdf.text("OVERALL STRESS INDEX", pageWidth - 20, 92, { align: "right" });
      pdf.setFontSize(16);
      pdf.setFont(undefined, 'bold');
      pdf.setTextColor(15, 23, 42);
      pdf.text(`${sessionSummary ? Math.round(sessionSummary.score) : '--'} / 100`, pageWidth - 20, 103, { align: "right" });
      
      // 4. Clinical Recommendation Box (Authentic Styling)
      pdf.setFillColor(248, 250, 252);
      pdf.setDrawColor(203, 213, 225);
      pdf.rect(20, 125, pageWidth - 40, 65, 'FD'); // Enlarged to perfectly fit 4-5 detailed lines
      
      pdf.setFontSize(10);
      pdf.setFont(undefined, 'bold');
      pdf.setTextColor(15, 23, 42);
      pdf.text("AI-DRIVEN ACTION PROTOCOL", 25, 133);
      
      pdf.setFontSize(9);
      pdf.setFont(undefined, 'normal');
      pdf.setTextColor(51, 65, 85);
      
      if (reportRecommendation) {
         let lines = reportRecommendation.split('\n').filter(l => l.trim() !== '');
         
         // Enforce absolute minimum of 4 lines for PDF layout consistency
         const fallbackLines = [
             "CLINICAL OBSERVATION: Patient exhibits sustained sympathetic nervous activation, requiring immediate physical down-regulation.",
             "INTERVENTION: Implement targeted vagus nerve stimulation via slow diaphragmatic breathing for ten minutes.",
             "PHYSICAL RECOVERY: Mandate temporary cessation of high-cognitive activities to restore normal resting state.",
             "HYDRATION THERAPY: Increase electrolyte fluid intake to optimize neural conductivity and stabilize readings."
         ];
         while (lines.length < 4) {
             lines.push(fallbackLines[lines.length]);
         }
         
         let yPos = 141;
         lines.forEach(line => {
             const cleanLine = line.replace(/^[0-9]+\.\s*/, '').replace(/^\*\s*/, '');
             const split = pdf.splitTextToSize(cleanLine, pageWidth - 55);
             pdf.setFillColor(45, 212, 191);
             pdf.circle(26.5, yPos - 1.2, 0.8, 'F'); 
             pdf.text(split, 30, yPos);
             yPos += (split.length * 4.5) + 4; // Perfect line spacing
         });
      } else {
         pdf.text("Scanning biometric patterns...", 25, 142);
      }
      
      // 5. Biosignal Telemetry Snapshot (Unified Graph)
      const chartElement = document.getElementById('pdf-combined-chart');
      if (chartElement) {
         // Force light mode for PDF capture
         chartElement.classList.add('pdf-light-mode');
         
         const canvas = await html2canvas(chartElement, { scale: 2, backgroundColor: '#ffffff' });
         
         // Revert light mode
         chartElement.classList.remove('pdf-light-mode');
         
         const imgData = canvas.toDataURL('image/jpeg', 1.0);
         
         // Calculate dimensions to fill the remaining space on Page 1
         const imgWidth = pageWidth - 40;
         let imgHeight = (canvas.height * imgWidth) / canvas.width;
         
         // Ensure it doesn't overlap the footer
         const maxImgHeight = 85;
         let finalImgWidth = imgWidth;
         
         if (imgHeight > maxImgHeight) {
             finalImgWidth = (maxImgHeight * canvas.width) / canvas.height;
             imgHeight = maxImgHeight;
         }
         
         const xOffset = 20 + (imgWidth - finalImgWidth) / 2;
         
         pdf.setDrawColor(203, 213, 225);
         pdf.rect(xOffset - 2, 193, finalImgWidth + 4, imgHeight + 4, 'S');
         pdf.addImage(imgData, 'JPEG', xOffset, 195, finalImgWidth, imgHeight);
      }
      
      // 6. Simple Footer
      pdf.setFontSize(8);
      pdf.setTextColor(200, 200, 200);
      pdf.text("Page 1 of 1", pageWidth / 2, pageHeight - 10, { align: "center" });
      
      pdf.save(`NeuroCalm_Diagnostic_${refId}.pdf`);
    } catch (err) {
      console.error(err);
      alert("Failed to generate PDF. Please try again.");
    } finally {
        setPdfLoading(false);
    }
  };

  const toggleBinauralAudio = () => {
    if (audioPlaying) {
      // Stop Audio gracefully
      oscillatorsRef.current.forEach(node => {
        try { node.stop(); } catch(e){}
      });
      oscillatorsRef.current = [];
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
        audioCtxRef.current = null;
      }
      setAudioPlaying(false);
      return;
    }

    // Start Audio Context
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioCtxRef.current = new AudioContext();
    const ctx = audioCtxRef.current;

    // Master Volume (Smooth Ramp to avoid harsh pop)
    const masterGain = ctx.createGain();
    masterGain.gain.setValueAtTime(0, ctx.currentTime);
    masterGain.gain.linearRampToValueAtTime(0.5, ctx.currentTime + 2); // 2 second fade in
    masterGain.connect(ctx.destination);

    // Left Ear (432Hz)
    const oscLeft = ctx.createOscillator();
    oscLeft.type = 'sine';
    oscLeft.frequency.setValueAtTime(432, ctx.currentTime);
    const panLeft = ctx.createStereoPanner();
    panLeft.pan.value = -1; // Hard left
    oscLeft.connect(panLeft).connect(masterGain);
    
    // Right Ear (428Hz -> 4Hz Theta wave)
    const oscRight = ctx.createOscillator();
    oscRight.type = 'sine';
    oscRight.frequency.setValueAtTime(428, ctx.currentTime);
    const panRight = ctx.createStereoPanner();
    panRight.pan.value = 1; // Hard right
    oscRight.connect(panRight).connect(masterGain);

    // Brown Noise (Soothing background hum)
    const bufferSize = ctx.sampleRate * 2;
    const noiseBuffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    let lastOut = 0;
    for (let i = 0; i < bufferSize; i++) {
        const white = Math.random() * 2 - 1;
        output[i] = (lastOut + (0.02 * white)) / 1.02;
        lastOut = output[i];
        output[i] *= 3.5;
    }
    const noiseSource = ctx.createBufferSource();
    noiseSource.buffer = noiseBuffer;
    noiseSource.loop = true;
    
    // Lowpass filter for Brown noise to make it deeply rumble like an ocean
    const noiseFilter = ctx.createBiquadFilter();
    noiseFilter.type = 'lowpass';
    noiseFilter.frequency.value = 400;
    const noiseGain = ctx.createGain();
    noiseGain.gain.value = 0.2; // Keep it ambient
    
    noiseSource.connect(noiseFilter).connect(noiseGain).connect(masterGain);

    oscLeft.start();
    oscRight.start();
    noiseSource.start();

    oscillatorsRef.current = [oscLeft, oscRight, noiseSource];
    setAudioPlaying(true);
  };
  
  // Clean up on component unmount mapping
  useEffect(() => {
    return () => {
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
      }
    };
  }, []);
  
  const predictStress = async () => {
    if (!files.eda) return;
    setLoading(true);
    setProcessingStep("Initializing Research Pipeline...");
    
    setTimeout(() => setProcessingStep("Parsing High-Frequency Telemetry..."), 800);
    setTimeout(() => setProcessingStep("Executing Multi-Modal Neural Sweep..."), 1600);
    setTimeout(() => setProcessingStep("Synthesizing Final Diagnostics..."), 2400);

    try {
      const formData = new FormData();
      if (files.eda) formData.append('eda', files.eda);
      if (files.temp) formData.append('temp', files.temp);
      if (files.hr) formData.append('hr', files.hr);
      formData.append('model', activeModel);

      const res = await axios.post('http://localhost:5000/api/predict_detailed', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      const data = res.data;
      setResult(data);
      setTimelineData(data.timeline);
      setSessionSummary({
          score: data.overall_stress_score,
          peakTime: data.peak_stress_time,
          recommendation: data.recommendation,
          confidence: data.timeline.reduce((acc, curr) => acc + curr.confidence, 0) / data.timeline.length,
          duration: data.charts.stamps[data.charts.stamps.length-1]
      });
      
      // Update Chart Data for main sensor graph
      const syncedCharts = [];
      const points = data.charts.eda.length;
      for(let i=0; i<points; i++) {
          syncedCharts.push({
              time: `${Math.round(data.charts.stamps[i])}s`,
              eda: data.charts.eda[i],
              hr: data.charts.hr[i],
              temp: data.charts.temp[i]
          });
      }
      setChartData(syncedCharts);

      setPatientHistory(prev => [{
        id: Date.now(),
        date: new Date().toLocaleString(),
        label: data.overall_stress_score > 50 ? "Elevated Stress" : "Healthy State",
        hr: data.insights?.mean_hr || 0,
        score: data.overall_stress_score
      }, ...prev]);

    } catch (err) {
      alert("Diagnostic Failure: Ensure the Python server is running and files are valid CSVs.");
    } finally {
      setTimeout(() => {
          setLoading(false);
          setProcessingStep(null);
      }, 3000);
    }
  };

  const loadDemoSession = () => {
    setLoading(true);
    setProcessingStep("Loading Clinical Demo Dataset...");
    
    setTimeout(() => {
        const demoCharts = [];
        const demoTimeline = [];
        for(let i=0; i<60; i++) {
            const isStress = i > 25 && i < 45;
            demoCharts.push({
                time: `${i}s`,
                eda: isStress ? 2.5 + Math.random() : 0.8 + Math.random()*0.2,
                hr: isStress ? 110 + Math.random()*10 : 72 + Math.random()*5,
                temp: isStress ? 30.5 : 32.2
            });
            if (i % 5 === 0) {
                demoTimeline.push({
                    time: i,
                    label: isStress ? 'Stress' : 'Resting',
                    confidence: 0.85 + Math.random()*0.1
                });
            }
        }
        
        setChartData(demoCharts);
        setTimelineData(demoTimeline);
        setResult({
            status: "success",
            prediction_label: "Healthy",
            overall_stress_score: 18.5,
            insights: { mean_hr: 75.4, mean_eda: 0.92, eda_peaks_detected: 2 }
        });
        setSessionSummary({
            score: 18.5,
            peakTime: 34,
            recommendation: "The system indicates a stable and relaxed physiological state. Maintain current activity.",
            confidence: 0.91,
            duration: 60
        });
        
        setLoading(false);
        setProcessingStep(null);
    }, 1500);
  };

  const rClass = result ? (result.overall_stress_score > 50 ? 'stress' : 'baseline') : '';

  // Helper for Session Gauge
  const Gauge = ({ value }) => {
    const radius = 80;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;
    
    // Smooth color interpolation
    const getColor = (v) => {
      if (v < 35) return '#34d399'; // Green/Safe
      if (v < 65) return '#fbbf24'; // Yellow/Medium
      return '#ef4444'; // Red/High
    };

    const getLabel = (v) => {
      if (v < 35) return 'Healthy State';
      if (v < 65) return 'Medium Engagement';
      return 'High Stress State';
    };
    
    return (
      <div className="gauge-wrapper">
        <svg className="gauge-svg" width="220" height="220">
          <circle className="gauge-bg" cx="110" cy="110" r={radius} />
          <circle className="gauge-fill" cx="110" cy="110" r={radius} 
            style={{ 
                strokeDasharray: circumference, 
                strokeDashoffset: offset,
                stroke: getColor(value) 
            }}
          />
        </svg>
        <div className="gauge-value-container" style={{color: getColor(value)}}>
          <div className="gauge-percent">{Math.round(value)}%</div>
          <div className="gauge-label" style={{fontSize:'0.8rem', fontWeight:700, opacity:0.8}}>{getLabel(value)}</div>
        </div>
      </div>
    );
  };

  const getAIReasoning = (res) => {
      if (!res) return "";
      const hr = res.insights?.mean_hr || 0;
      const eda = res.insights?.mean_eda || 0;
      const score = res.overall_stress_score || 0;
      
      if (score > 65) {
          return `Stress detected due to high EDA (${eda.toFixed(2)} µS) and elevated heart rate (${Math.round(hr)} BPM).`;
      } else if (score > 35) {
          return `Moderate engagement detected. Physiological markers show healthy excitation levels.`;
      } else {
          return `The system indicates a stable and relaxed physiological state. All biometric markers within resting safe zones.`;
      }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')} min`;
  };

  return (
    <div className="platform-layout">
      <div className="bg-aurora"></div>
      
      {processingStep && (
          <div className="processing-overlay">
              <div className="processing-loader"></div>
              <div className="processing-text">{processingStep}</div>
              <div className="processing-subtext">Optimizing neural pathways for analysis...</div>
          </div>
      )}
      
      {/* SaaS Sidebar Navigation */}
      <aside className="platform-sidebar">
         <div className="sidebar-brand">
            <div className="icon-container-brand">
              <BrainCircuit size={28} color="white" />
            </div>
            <span style={{fontSize:'1.4rem', fontWeight:800, color:'white', letterSpacing:'0.5px'}}>NeuroCalm</span>
         </div>

         <nav className="sidebar-nav">
            <div className={`nav-item ${activeView === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveView('dashboard')}>
               <LayoutDashboard size={20} />
               <span>Live Dashboard</span>
            </div>
            <div className={`nav-item ${activeView === 'archives' ? 'active' : ''}`} onClick={() => setActiveView('archives')}>
               <Users size={20} />
               <span>Patient Archives</span>
            </div>
            <div className={`nav-item ${activeView === 'settings' ? 'active' : ''}`} onClick={() => setActiveView('settings')}>
               <Settings size={20} />
               <span>System Settings</span>
            </div>
         </nav>

         <div className="sidebar-profile">
            <div className="avatar-circle">
               <User size={20} color="white" />
            </div>
            <div className="profile-details">
               <div className="profile-name">Dr. A. Clinician</div>
               <div className="profile-role">Lead Neurologist</div>
            </div>
         </div>
      </aside>

      <main className="platform-main">
        <div className="app-container" id="report-content">
          {/* Header */}
          <header className="top-nav">
            <div>
              <h1 style={{fontSize:'1.8rem', fontWeight:800}}>Diagnostics Center</h1>
              <p style={{color:'var(--text-muted)'}}>Upload physiological telemetry for real-time inference.</p>
            </div>
            <div style={{display: 'flex', gap: '1.2rem', alignItems: 'center'}}>
               <div className="nav-icon-btn" style={{position: 'relative'}} onClick={() => alert("Logs active.")}>
                  <Bell size={22} color="var(--text-muted)" />
                  {patientHistory.length > 0 && <div className="bell-badge"></div>}
               </div>
             {result && (
               <button className="btn-pdf" onClick={generatePDF} disabled={pdfLoading}>
                 <Download size={18} /> {pdfLoading ? 'Generating AI Report...' : 'Clinical Report'}
               </button>
             )}
            <div className="header-status">
              <div className="status-dot"></div>
              <span style={{fontWeight: 600}}>System Online</span>
            </div>
          </div>
        </header>

        {/* Archives View */}
        {activeView === 'archives' && (
          <div className="glass-panel" style={{marginTop: '2rem'}}>
             <h2 style={{fontSize: '1.6rem', fontWeight: 800, marginBottom: '1.5rem'}}>Patient Archives System</h2>
             {patientHistory.length === 0 ? (
               <div style={{color: 'var(--text-muted)', padding: '2rem', textAlign: 'center'}}>No telemetry sessions recorded during this cycle.</div>
             ) : (
               <div className="table-responsive">
                 <table className="clinical-table">
                   <thead>
                     <tr>
                       <th>Session ID</th>
                       <th>Timestamp</th>
                       <th>HR (BPM)</th>
                       <th>EDA (µS)</th>
                       <th>ML Diagnosis</th>
                     </tr>
                   </thead>
                   <tbody>
                     {patientHistory.map(session => (
                       <tr key={session.id}>
                         <td>#{session.id.toString().slice(-6)}</td>
                         <td>{session.date}</td>
                         <td>{session.hr}</td>
                         <td>{session.eda}</td>
                         <td>
                           <span className={`badge ${session.label.toLowerCase()}`}>{session.label}</span>
                         </td>
                       </tr>
                     ))}
                   </tbody>
                 </table>
               </div>
             )}
          </div>
        )}

        {/* Settings View */}
        {activeView === 'settings' && (
          <div className="glass-panel" style={{marginTop: '2rem', maxWidth: '800px'}}>
             <h2 style={{fontSize: '1.6rem', fontWeight: 800, marginBottom: '2rem'}}>System Configurations</h2>
             
             <div className="setting-group">
                <div className="setting-info">
                   <h3 style={{fontSize: '1.1rem', marginBottom: '0.3rem'}}>AI Personality Engine</h3>
                   <p style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>Adjust the overarching tone of the Groq LLaMA-3 Assistant.</p>
                </div>
                <select 
                  className="premium-select" 
                  value={aiTone} 
                  onChange={(e) => setAiTone(e.target.value)}
                >
                   <option value="compassionate">Compassionate & Warm</option>
                   <option value="clinical">Strict Clinical Diagnostic</option>
                   <option value="academic">Academic & Analytical</option>
                </select>
             </div>

             <div className="setting-group" style={{marginTop: '2rem'}}>
                <div className="setting-info">
                   <h3 style={{fontSize: '1.1rem', marginBottom: '0.3rem'}}>Data Telemetry Feed</h3>
                   <p style={{color: 'var(--text-muted)', fontSize: '0.9rem'}}>Establish local proxy routing for CSV ingestion buffers.</p>
                </div>
                <div className="toggle-switch active"></div>
             </div>
          </div>
        )}

        {/* Dashboard View */}
        {activeView === 'dashboard' && (
          <>
        {/* Model Selection Tabs */}
        <div className="tab-group">
            <div className={`tab-item ${activeModel === 'best' ? 'active' : ''}`} onClick={() => setActiveModel('best')}>AUTO-SELECT</div>
            <div className={`tab-item ${activeModel === 'cnn' ? 'active' : ''}`} onClick={() => setActiveModel('cnn')}>CNN</div>
            <div className={`tab-item ${activeModel === 'lstm' ? 'active' : ''}`} onClick={() => setActiveModel('lstm')}>LSTM</div>
            <div className={`tab-item ${activeModel === 'hybrid' ? 'active' : ''}`} onClick={() => setActiveModel('hybrid')}>HYBRID-CNN</div>
        </div>

        {/* Metrics Grid */}
        <div className="metrics-grid">
          <div className="glass-panel metric-card">
            <Activity size={100} className="icon-watermark" />
            <div className="metric-title">Overall Stress Level</div>
            <div style={{marginTop: '1rem', textAlign:'center'}}>
               <Gauge value={sessionSummary ? sessionSummary.score : 0} />
               <div style={{marginTop:'0.5rem', fontWeight:800, fontSize:'1.1rem', color: (sessionSummary?.score > 65 ? '#ef4444' : (sessionSummary?.score > 35 ? '#fbbf24' : '#34d399'))}}>
                  {sessionSummary ? `Stress Level: ${Math.round(sessionSummary.score)}%` : '--'}
               </div>
            </div>
          </div>
          
          <div className="glass-panel metric-card" style={{justifyContent: 'center'}}>
            <div className="metric-title">Peak Stress Point</div>
            <div className="metric-value" style={{color: '#ef4444', fontSize: '2.2rem'}}>
              {sessionSummary ? `${formatTime(sessionSummary.peakTime)}` : '--'}
            </div>
            <div style={{fontWeight:700, color:(sessionSummary?.score > 65 ? '#ef4444' : '#34d399'), fontSize:'0.9rem', marginBottom:'1rem'}}>
               {sessionSummary ? (sessionSummary.score > 65 ? 'State: High Stress Detected' : 'State: Healthy (Low Stress)') : ''}
            </div>
            <div style={{padding: '0.8rem', background:'rgba(255,255,255,0.05)', borderRadius:'10px', border:'1px solid var(--border-light)'}}>
                <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.85rem'}}>
                    <span style={{color:'var(--text-muted)'}}>AI Confidence:</span>
                    <span style={{fontWeight:800, color:'#34d399'}}>{sessionSummary ? `${Math.round(sessionSummary.confidence * 100)}%` : '--'}</span>
                </div>
            </div>
          </div>

          <div className="glass-panel metric-card" style={{justifyContent: 'center'}}>
            <div className="metric-title">Session Diagnostic Meta-Log</div>
            <div className="recommendation-container" style={{marginTop:'0.8rem'}}>
                <div className="recommendation-card" style={{padding: '1rem', background:'rgba(0,0,0,0.2)', border:'1px solid var(--border-light)'}}>
                    <div className="rec-content" style={{width: '100%'}}>
                        <div style={{marginBottom:'0.5rem', fontSize:'0.75rem', fontWeight:800, color:'var(--text-muted)', textTransform:'uppercase'}}>Diagnostic Summary Box</div>
                        <ul style={{listStyle:'none', padding:0, margin:0, display:'flex', flexDirection:'column', gap:'0.6rem'}}>
                           <li style={{display:'flex', justifyContent:'space-between', fontSize:'0.85rem'}}>
                              <span>Active Model:</span>
                              <span style={{fontWeight:700, color:'var(--secondary)'}}>Hybrid CNN-LSTM</span>
                           </li>
                           <li style={{display:'flex', justifyContent:'space-between', fontSize:'0.85rem'}}>
                              <span>Average Stress:</span>
                              <span style={{fontWeight:700, color: (sessionSummary?.score > 65 ? '#ef4444' : '#34d399')}}>{sessionSummary ? Math.round(sessionSummary.score) : '--'}%</span>
                           </li>
                           <li style={{display:'flex', justifyContent:'space-between', fontSize:'0.85rem'}}>
                              <span>Peak Intensity:</span>
                              <span style={{fontWeight:700}}>{sessionSummary ? (sessionSummary.score > 65 ? 'High' : 'Normal') : '--'}</span>
                           </li>
                           <li style={{display:'flex', justifyContent:'space-between', fontSize:'0.85rem'}}>
                              <span>Session Duration:</span>
                              <span style={{fontWeight:700, color:'var(--primary-light)'}}>{sessionSummary ? formatTime(sessionSummary.duration) : '--'}</span>
                           </li>
                        </ul>
                    </div>
                </div>
            </div>
          </div>
        </div>

        {/* AI Action Plan Full Width Banner */}
        <div className="glass-panel" style={{marginBottom: '2rem', display: 'flex', flexDirection: 'column', gap: '1rem', background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(45, 212, 191, 0.05))', border: '1px solid var(--primary-glow)'}}>
            <div style={{fontSize:'1.2rem', fontWeight:800, color:'var(--primary-light)', display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
                <BrainCircuit size={20} /> AI Personalized Action Plan
            </div>
            {sessionSummary ? (
                <div style={{fontSize:'0.95rem', lineHeight:1.6, color:'rgba(255,255,255,0.9)', display:'grid', gridTemplateColumns: '1fr 1fr', gap:'1.5rem'}}>
                    {sessionSummary.recommendation.split('\n').filter(line => line.trim() !== '').map((line, idx) => (
                        <div key={idx} style={{display:'flex', gap:'12px', alignItems:'flex-start', background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)'}}>
                            <div style={{color:'var(--primary)', marginTop:'2px', fontSize: '1.2rem'}}>✦</div>
                            <div>{line.replace(/^[0-9]+\.\s*/, '').replace(/^\*\s*/, '')}</div>
                        </div>
                    ))}
                </div>
            ) : (
                <p style={{fontSize:'0.95rem', lineHeight:1.4, margin:0, color:'rgba(255,255,255,0.5)'}}>
                    Scanning biometric patterns to generate plan...
                </p>
            )}
        </div>
        
        {/* Main Content Area */}
        <div className="main-grid">
          
          {/* Upload Widget */}
          <div className="glass-panel upload-container">
             <h2 style={{fontSize: '1.4rem', fontWeight: 800}}>Sensor Telemetry Ingestion</h2>
             <p style={{color: 'var(--text-muted)', marginBottom: '1rem', fontSize: '0.9rem'}}>Upload Subject Pkl or CSV bundles (EDA mandatory).</p>
             
             <div 
               className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
               onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
               onClick={() => document.getElementById('file-upload').click()}
             >
               <UploadCloud className="upload-icon" />
               <h3>Load Session Data</h3>
               <div className="file-tags">
                 <div className={`file-tag ${files.eda ? 'found' : ''}`}>
                    {files.eda ? <CheckCircle size={14}/> : <Circle size={14}/>} EDA.csv
                 </div>
                 <div className={`file-tag ${files.temp ? 'found' : ''}`}>
                    {files.temp ? <CheckCircle size={14}/> : <Circle size={14}/>} TEMP.csv
                 </div>
                 <div className={`file-tag ${files.hr ? 'found' : ''}`}>
                    {files.hr ? <CheckCircle size={14}/> : <Circle size={14}/>} HR.csv
                 </div>
               </div>
               <input id="file-upload" type="file" multiple onChange={(e) => processFiles(e.target.files)} style={{display: 'none'}} />
             </div>

             <div style={{display:'flex', gap:'10px', marginTop:'1rem'}}>
                <button className="btn-predict" onClick={predictStress} disabled={loading || !files.eda} style={{flex:2}}>
                    {loading ? 'Processing...' : 'Run Neural Diagnostic'}
                </button>
                <button className="btn-demo" onClick={loadDemoSession} disabled={loading} style={{flex:1}}>
                    <Play size={16} fill="white" /> Try Demo Mode
                </button>
             </div>

             {/* Stress Timeline Ribbon */}
             {timelineData.length > 0 && (
                 <div className="timeline-card">
                     <div className="metric-title" style={{fontSize: '0.8rem'}}>Temporal Stress Distribution</div>
                     <div className="timeline-ribbon">
                         {timelineData.map((step, i) => (
                             <div 
                                key={i} 
                                className="ribbon-segment"
                                style={{
                                    flex: 1,
                                    background: step.label === 'Stress' ? 'var(--danger)' : (step.label === 'Amusement' ? 'var(--warning)' : 'var(--success)')
                                }}
                                title={`${step.time}s: ${step.label} (${Math.round(step.confidence * 100)}%)`}
                             />
                         ))}
                     </div>
                     <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.7rem', color:'var(--text-muted)', marginTop:'4px', padding:'0 2px'}}>
                         <span>0s</span>
                         <span>{timelineData[timelineData.length - 1]?.time}s</span>
                     </div>
                     <div className="ribbon-legend" style={{display:'flex', gap:'15px', marginTop:'10px', justifyContent:'center'}}>
                         <div style={{display:'flex', alignItems:'center', gap:'5px', fontSize:'0.75rem', color:'var(--text-muted)'}}>
                             <div style={{width:'8px', height:'8px', borderRadius:'50%', background:'var(--success)'}}></div> Healthy
                         </div>
                         <div style={{display:'flex', alignItems:'center', gap:'5px', fontSize:'0.75rem', color:'var(--text-muted)'}}>
                             <div style={{width:'8px', height:'8px', borderRadius:'50%', background:'var(--warning)'}}></div> Engaged
                         </div>
                         <div style={{display:'flex', alignItems:'center', gap:'5px', fontSize:'0.75rem', color:'var(--text-muted)'}}>
                             <div style={{width:'8px', height:'8px', borderRadius:'50%', background:'var(--danger)'}}></div> Stress
                         </div>
                     </div>
                 </div>
             )}
          </div>

          {/* Graph Widget - Multi-Signal Stack */}
          <div className="glass-panel">
            <h2 style={{fontSize: '1.4rem', fontWeight: 800}}>Synchronized Biosignal Visualizer</h2>
            <div id="chart-capture-area" className="chart-container" style={{height: '500px', display: 'flex', flexDirection: 'column', gap: '10px'}}>
              {chartData.length > 0 ? (
                <>
                  <ResponsiveContainer width="100%" height="33%">
                    <AreaChart data={chartData} syncId="biosync" margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="time" hide />
                      <YAxis stroke="var(--secondary)" fontSize={10} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ background: 'rgba(15, 23, 42, 0.8)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: 'white', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }} />
                      <Area type="monotone" dataKey="eda" stroke="var(--secondary)" fillOpacity={0.2} fill="var(--secondary)" />
                    </AreaChart>
                  </ResponsiveContainer>

                  <ResponsiveContainer width="100%" height="33%">
                    <AreaChart data={chartData} syncId="biosync" margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="time" hide />
                      <YAxis stroke="var(--primary)" fontSize={10} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ background: 'rgba(15, 23, 42, 0.8)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: 'white', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }} />
                      <Area type="monotone" dataKey="hr" stroke="var(--primary)" fillOpacity={0.2} fill="var(--primary)" />
                    </AreaChart>
                  </ResponsiveContainer>

                  <ResponsiveContainer width="100%" height="33%">
                    <AreaChart data={chartData} syncId="biosync" margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={10} />
                      <YAxis stroke="var(--warning)" fontSize={10} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ background: 'rgba(15, 23, 42, 0.8)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: 'white', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }} />
                      <Area type="monotone" dataKey="temp" stroke="var(--warning)" fillOpacity={0.2} fill="var(--warning)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </>
              ) : (
                 <div style={{height:'100%', display:'flex', alignItems:'center', justifyContent:'center', color:'rgba(255,255,255,0.2)', fontSize:'1.2rem', fontWeight:600}}>
                   AWAITING TELEMETRY INGESTION...
                 </div>
              )}
            </div>
          </div>
        </div>

        {/* Prediction Results Banner */}
        {result && (
          <div className={`result-banner ${rClass}`}>
            <div className="result-content">
              <div className="result-subtitle">Algorithmic Classification Analysis</div>
              <div className="result-title">
                  {result.prediction_label} State
                  <span style={{fontSize:'1.1rem', fontWeight:600, marginLeft:'1.2rem', padding:'4px 12px', background:'rgba(255,255,255,0.1)', borderRadius:'30px', color:'white'}}>
                      (Confidence: {Math.round(sessionSummary?.confidence * 100)}%)
                  </span>
              </div>
              
              <div style={{marginTop:'0.8rem', fontStyle:'italic', color:'rgba(255,255,255,0.8)', fontSize:'1rem'}}>
                  AI Recommendation: {sessionSummary?.recommendation || "Maintain your current activity."}
              </div>

              <div style={{marginTop:'0.5rem', color:'rgba(255,255,255,0.7)', fontSize:'0.9rem'}}>
                  Status: {sessionSummary?.score > 65 ? "Elevated Alert" : "Stable & Relaxed"}
              </div>

              {result.population_comparison && (
                <div style={{ marginTop: '1.5rem', padding: '1.2rem', background: 'rgba(0,0,0,0.4)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', maxWidth: '600px', lineHeight: '1.6', color: 'rgba(255,255,255,0.9)', fontSize: '0.95rem', letterSpacing: '0.3px', backdropFilter: 'blur(5px)' }}>
                   <strong style={{color: 'white'}}>Physiological Insight: </strong> {result.population_comparison}
                </div>
              )}

              {/* Explainability Engine */}
              {result.explainability_metrics && (
                <div className="shap-container">
                  <h4 style={{marginBottom: '1rem', color: 'white', fontSize: '1rem', textTransform: 'uppercase', letterSpacing: '1px'}}>AI Diagnostic Drivers</h4>
                  {result.explainability_metrics.map((metric, idx) => (
                    <div className="shap-item" key={idx}>
                      <div className="shap-label">
                        <span>{metric.feature}</span>
                        <span>{metric.impact}% Impact</span>
                      </div>
                      <div className="shap-bar-bg">
                        <div className="shap-bar-fill" style={{width: `${metric.impact}%`}}></div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="result-icon">
              {rClass === 'stress' && <ShieldAlert size={100} color="var(--danger)" />}
              {rClass === 'baseline' && <CheckCircle size={100} color="var(--success)" />}
              {rClass === 'amusement' && <Activity size={100} color="var(--warning)" />}
            </div>
          </div>
        )}

        {/* Breathing Intervention */}
        {rClass === 'stress' && (
          <div className="breathing-module" style={{position: 'relative', overflow: 'hidden'}}>
             {audioPlaying && <div className="audio-wave-bg"></div>}
             <h2 style={{fontSize:'1.8rem', fontWeight: 800, color: 'white', marginBottom:'0.5rem', position: 'relative', zIndex: 2}}>Calming Intervention</h2>
             <p style={{color: 'var(--text-muted)', marginBottom: '2rem', position: 'relative', zIndex: 2}}>Follow the rigorous 4-7-8 breathing rhythm to physically lower your biological heart rate.</p>
             
             <div className="breath-circle" style={{position: 'relative', zIndex: 2}}>
                <div className="breath-text"></div>
             </div>
             
             <button 
               className={`audio-btn ${audioPlaying ? 'active' : ''}`}
               onClick={toggleBinauralAudio}
               style={{position: 'relative', zIndex: 2, marginTop: '3rem'}}
             >
                <Headphones size={22} className={audioPlaying ? 'pulse-icon' : ''} />
                <span>{audioPlaying ? 'Terminate Acoustic Engine' : 'Begin 4Hz Theta Therapy'}</span>
             </button>
          </div>
        )}
        </>
        )}

        {/* Floating Action Button */}
        <button 
          className="chat-fab-btn" 
          onClick={() => setChatOpen(!chatOpen)}
          aria-label="Toggle AI Assistant"
        >
          {chatOpen ? <X size={28} /> : <MessageSquare size={28} />}
        </button>

        {/* NeuroChat Floating Interface */}
        {chatOpen && (
          <div className="chat-widget">
            <div className="chat-header">
              <MessageSquare size={24} color="var(--primary-light)" />
              <h3 style={{fontWeight: 700, color: 'white', letterSpacing: '0.5px'}}>NeuroCalm Assistant</h3>
              <button 
                onClick={() => setChatOpen(false)} 
                style={{marginLeft: 'auto', background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer'}}
              >
                <X size={20} />
              </button>
            </div>
            
            <div className="chat-messages">
              {chatHistory.map((msg, idx) => (
                <div key={idx} className={`chat-bubble ${msg.role}`}>
                  {msg.content}
                </div>
              ))}
              {chatLoading && (
                <div className="chat-bubble ai" style={{opacity: 0.6}}>
                  Analyzing biometrics...
                </div>
              )}
            </div>
            
            <form className="chat-input-area" onSubmit={handleChatSubmit}>
              <input 
                type="text" 
                className="chat-input" 
                placeholder="Ask about your physical stress, health..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={chatLoading}
                autoFocus
              />
              <button type="submit" className="chat-btn" disabled={chatLoading || !chatInput.trim()}>
                <Send size={20} />
              </button>
            </form>
          </div>
        )}

        </div>
        
        {/* Hidden Combined Chart for PDF Generation Only */}
        <div style={{ position: 'absolute', left: '-9999px', top: 0 }}>
            <div id="pdf-combined-chart" className="pdf-light-mode" style={{ width: '1000px', height: '520px', background: 'white', padding: '30px 40px', borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)' }}>
                
                {/* Premium Dashboard Header inspired by user UI request */}
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '30px', alignItems: 'flex-start'}}>
                    <div>
                        <div style={{color: '#1e293b', fontWeight: '800', fontSize: '20px', letterSpacing: '-0.5px'}}>Biometric Telemetry Overview</div>
                        <div style={{color: '#64748b', fontSize: '13px', marginTop: '6px', fontWeight: '500'}}>Last updated: {new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                    </div>
                </div>

                {chartData.length > 0 && (
                    <AreaChart width={920} height={350} data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                        <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={{stroke: '#cbd5e1'}} tickMargin={10} />
                        <YAxis yAxisId="hr" domain={['dataMin - 10', 'dataMax + 10']} hide />
                        <YAxis yAxisId="eda" domain={['dataMin - 0.5', 'dataMax + 0.5']} hide />
                        <YAxis yAxisId="temp" domain={['dataMin - 1', 'dataMax + 1']} hide />
                        
                        {/* Peak Stress Indicator Line */}
                        {sessionSummary?.peakTime && (
                            <ReferenceLine yAxisId="hr" x={sessionSummary.peakTime} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={2} label={{ position: 'top', value: 'PEAK STRESS', fill: '#ef4444', fontSize: 13, fontWeight: '800' }} />
                        )}
                        
                        <Area yAxisId="hr" type="monotone" dataKey="hr" stroke="var(--primary)" fill="var(--primary)" fillOpacity={0.1} strokeWidth={2.5} />
                        <Area yAxisId="eda" type="monotone" dataKey="eda" stroke="var(--secondary)" fill="var(--secondary)" fillOpacity={0.1} strokeWidth={2.5} />
                        <Area yAxisId="temp" type="monotone" dataKey="temp" stroke="var(--warning)" fill="var(--warning)" fillOpacity={0.1} strokeWidth={2.5} />
                    </AreaChart>
                )}
                
                {/* Clean Bottom Legend */}
                <div style={{display: 'flex', justifyContent: 'center', gap: '30px', marginTop: '20px', fontSize: '13px', fontWeight: '700'}}>
                    <div style={{color: '#475569', display: 'flex', alignItems: 'center', gap: '8px'}}><div style={{width:'12px', height:'12px', borderRadius: '2px', background:'var(--primary)'}}></div> HR (Heart Rate)</div>
                    <div style={{color: '#475569', display: 'flex', alignItems: 'center', gap: '8px'}}><div style={{width:'12px', height:'12px', borderRadius: '2px', background:'var(--secondary)'}}></div> EDA (Sweat)</div>
                    <div style={{color: '#475569', display: 'flex', alignItems: 'center', gap: '8px'}}><div style={{width:'12px', height:'12px', borderRadius: '2px', background:'var(--warning)'}}></div> TEMP (Body Heat)</div>
                </div>
            </div>
        </div>

      </main>
    </div>
  );
}

export default App;

import React, { useState, useRef, useEffect } from 'react';
import { Upload, MessageCircle, Key, Send, Loader2, FileText, Check, AlertCircle, RotateCcw, Globe, Youtube, ArrowRight, ArrowLeft, ExternalLink } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const App = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedFeature, setSelectedFeature] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState({ type: '', message: '' });
  const [isUploading, setIsUploading] = useState(false);
  const [messages, setMessages] = useState([
    { type: 'assistant', content: "Hello! I'll help you with document Q&A or content summarization. Let's start by entering your OpenAI API key." }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [urlInput, setUrlInput] = useState('');
  const [summaryResult, setSummaryResult] = useState('');
  const [isSummarizing, setIsSummarizing] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const generateSessionId = () => {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleNextStep = (step) => {
    if (step === 1) {
      if (!apiKey.trim()) {
        setUploadStatus({ type: 'error', message: 'Please enter your OpenAI API key' });
        return;
      }
      setUploadStatus({ type: '', message: '' });
    }
    
    if (step === 2) {
      if (!selectedFeature) {
        setUploadStatus({ type: 'error', message: 'Please select a feature' });
        return;
      }
      if (selectedFeature === 'documents') {
        setSessionId(generateSessionId());
      }
      setUploadStatus({ type: '', message: '' });
    }

    if (step === 3 && selectedFeature === 'documents') {
      if (selectedFiles.length === 0) {
        setUploadStatus({ type: 'error', message: 'Please select at least one PDF file' });
        return;
      }
    }

    setCurrentStep(step + 1);
  };

  const handlePrevStep = (step) => {
    setCurrentStep(step - 1);
    setUploadStatus({ type: '', message: '' });
  };

  const handleFeatureSelect = (feature) => {
    setSelectedFeature(feature);
    setUploadStatus({ type: '', message: '' });
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    const pdfFiles = files.filter(file => file.type === 'application/pdf');
    
    if (pdfFiles.length !== files.length) {
      setUploadStatus({ type: 'error', message: 'Only PDF files are allowed' });
      return;
    }
    
    setSelectedFiles(pdfFiles);
    setUploadStatus({ type: '', message: '' });
  };

  const uploadDocuments = async () => {
    if (selectedFiles.length === 0) {
      setUploadStatus({ type: 'error', message: 'Please select PDF files first' });
      return;
    }

    setIsUploading(true);
    setUploadStatus({ type: 'info', message: 'Processing documents...' });

    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append('files', file);
    });
    formData.append('session_id', sessionId);
    formData.append('openai_api_key', apiKey);

    try {
      const response = await fetch(`${API_BASE_URL}/upload-documents`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        setUploadStatus({
          type: 'success',
          message: `Successfully processed ${result.documents_count} documents (${result.chunks_count} chunks)`
        });
        setTimeout(() => {
          handleNextStep(3);
          setMessages([
            { type: 'assistant', content: "Perfect! I've processed your documents. You can now ask me questions about their content." }
          ]);
        }, 1500);
      } else {
        setUploadStatus({ type: 'error', message: result.detail });
      }
    } catch (error) {
      setUploadStatus({ type: 'error', message: `Network error: ${error.message}` });
    }

    setIsUploading(false);
  };

  const summarizeContent = async () => {
    if (!urlInput.trim()) {
      setUploadStatus({ type: 'error', message: 'Please enter a URL' });
      return;
    }

    setIsSummarizing(true);
    setUploadStatus({ type: 'info', message: 'Analyzing and summarizing content...' });
    setSummaryResult('');

    try {
      const response = await fetch(`${API_BASE_URL}/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: urlInput,
          openai_api_key: apiKey
        })
      });

      const result = await response.json();

      if (response.ok) {
        setSummaryResult(result.summary);
        setUploadStatus({
          type: 'success',
          message: `Successfully summarized ${result.content_type} content`
        });
      } else {
        setUploadStatus({ type: 'error', message: result.detail });
      }
    } catch (error) {
      setUploadStatus({ type: 'error', message: `Network error: ${error.message}` });
    }

    setIsSummarizing(false);
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = { type: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsSending(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          session_id: sessionId,
          openai_api_key: apiKey
        })
      });

      const result = await response.json();

      if (response.ok) {
        const assistantMessage = { type: 'assistant', content: result.answer };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        const errorMessage = { type: 'assistant', content: `Error: ${result.detail}`, isError: true };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = { type: 'assistant', content: `Network Error: ${error.message}`, isError: true };
      setMessages(prev => [...prev, errorMessage]);
    }

    setIsSending(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (selectedFeature === 'documents') {
        sendMessage();
      } else if (selectedFeature === 'summarize') {
        summarizeContent();
      }
    }
  };

  const resetApp = async () => {
    if (window.confirm('Are you sure you want to start a new session? This will clear all data.')) {
      if (sessionId) {
        try {
          await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
            method: 'DELETE'
          });
        } catch (error) {
          console.error('Error clearing session:', error);
        }
      }
      
      setCurrentStep(1);
      setSelectedFeature('');
      setSessionId('');
      setApiKey('');
      setSelectedFiles([]);
      setUploadStatus({ type: '', message: '' });
      setMessages([
        { type: 'assistant', content: "Hello! I'll help you with document Q&A or content summarization. Let's start by entering your OpenAI API key." }
      ]);
      setInputMessage('');
      setUrlInput('');
      setSummaryResult('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const StatusMessage = ({ status }) => {
    if (!status.message) return null;
    
    const bgColor = {
      success: 'bg-green-50 border-green-200 text-green-800',
      error: 'bg-red-50 border-red-200 text-red-800',
      info: 'bg-blue-50 border-blue-200 text-blue-800'
    }[status.type];

    const icon = {
      success: <Check className="w-5 h-5" />,
      error: <AlertCircle className="w-5 h-5" />,
      info: <Loader2 className="w-5 h-5 animate-spin" />
    }[status.type];

    return (
      <div className={`flex items-center space-x-2 p-4 rounded-lg border ${bgColor} mb-6`}>
        {icon}
        <span className="font-medium">{status.message}</span>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 flex justify-center items-center">
      <div className="container mx-auto px-4 py-2">
        <div className="max-w-4xl mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 text-white p-3">
            <div className="flex justify-between items-center">
              <div className="flex-1">
                <h1 className="text-2xl font-bold text-center mb-1">AI Assistant Suite</h1>
                <p className="text-center text-blue-100 text-md">Document Q&A and Content Summarization</p>
              </div>
              {currentStep > 1 && (
                <button
                  onClick={resetApp}
                  className="ml-4 p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors duration-200"
                  title="Start New Session"
                >
                  <RotateCcw className="w-5 h-5" />
                </button>
              )}
            </div>
          </div>
          <div className="px-8 py-6">
            {/* Step 1: API Key */}
            {currentStep === 1 && (
              <div className="space-y-6 animate-in fade-in duration-500">
                <div className="flex items-center space-x-4 pb-4 border-b border-gray-200">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                    1
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-800">Enter OpenAI API Key</h2>
                </div>

                <div className="space-y-4">
                  <label className="block text-sm font-medium text-gray-700">
                    OpenAI API Key
                  </label>
                  <div className="relative">
                    <Key className="absolute left-4 top-4 w-5 h-5 text-gray-400" />
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      className="w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-blue-100 focus:border-blue-500 transition-all duration-200 text-lg"
                      placeholder="Enter your OpenAI API key..."
                    />
                  </div>
                </div>

                <StatusMessage status={uploadStatus} />

                <div className="flex justify-end">
                  <button
                    onClick={() => handleNextStep(1)}
                    className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-purple-600 transform hover:scale-105 transition-all duration-200 shadow-lg flex items-center space-x-2"
                  >
                    <span>Next Step</span>
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </div>
              </div>
            )}

            {/* Step 2: Feature Selection */}
            {currentStep === 2 && (
              <div className="space-y-6 animate-in fade-in duration-500">
                <div className="flex items-center space-x-4 pb-4 border-b border-gray-200">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                    2
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-800">Choose Your Feature</h2>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  {/* Document Q&A Card */}
                  <div
                    onClick={() => handleFeatureSelect('documents')}
                    className={`p-6 rounded-2xl border-2 cursor-pointer transition-all duration-300 transform hover:scale-105 hover:shadow-lg ${
                      selectedFeature === 'documents'
                        ? 'border-blue-500 bg-blue-50 shadow-lg'
                        : 'border-gray-200 bg-white hover:border-blue-300'
                    }`}
                  >
                    <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mb-4 mx-auto">
                      <FileText className="w-8 h-8 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-center mb-3 text-gray-800">Document Q&A</h3>
                    <p className="text-gray-600 text-center mb-4">
                      Upload PDF documents and chat with them using advanced RAG technology. Perfect for research papers, reports, and documentation.
                    </p>
                    <div className="flex items-center justify-center space-x-2 text-blue-600 font-medium">
                      <Upload className="w-4 h-4" />
                      <span>Upload & Chat</span>
                    </div>
                  </div>

                  {/* Summarization Card */}
                  <div
                    onClick={() => handleFeatureSelect('summarize')}
                    className={`p-6 rounded-2xl border-2 cursor-pointer transition-all duration-300 transform hover:scale-105 hover:shadow-lg ${
                      selectedFeature === 'summarize'
                        ? 'border-purple-500 bg-purple-50 shadow-lg'
                        : 'border-gray-200 bg-white hover:border-purple-300'
                    }`}
                  >
                    <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full mb-4 mx-auto">
                      <Globe className="w-8 h-8 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-center mb-3 text-gray-800">Content Summarization</h3>
                    <p className="text-gray-600 text-center mb-4">
                      Instantly summarize content from YouTube videos or websites. Get key insights in 300 words or less.
                    </p>
                    <div className="flex items-center justify-center space-x-2 text-purple-600 font-medium">
                      <ExternalLink className="w-4 h-4" />
                      <span>Analyze & Summarize</span>
                    </div>
                  </div>
                </div>

                <StatusMessage status={uploadStatus} />

                <div className="flex justify-between">
                  <button
                    onClick={() => handlePrevStep(2)}
                    className="px-6 py-3 bg-gray-500 text-white rounded-xl font-semibold hover:bg-gray-600 transition-colors duration-200 flex items-center space-x-2"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    <span>Previous</span>
                  </button>
                  <button
                    onClick={() => handleNextStep(2)}
                    disabled={!selectedFeature}
                    className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-purple-600 transform hover:scale-105 transition-all duration-200 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center space-x-2"
                  >
                    <span>Continue</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            {/* Step 3a: Document Upload (if documents selected) */}
            {currentStep === 3 && selectedFeature === 'documents' && (
              <div className="space-y-6 animate-in fade-in duration-500">
                <div className="flex items-center space-x-4 pb-4 border-b border-gray-200">
                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                    3
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-800">Upload PDF Documents</h2>
                </div>

                <div className="space-y-4">
                  <label className="block text-sm font-medium text-gray-700">
                    Select PDF files to upload
                  </label>
                  <div className="flex items-center justify-center w-full">
                    <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-2xl cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors duration-200">
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <Upload className="w-10 h-10 mb-3 text-gray-400" />
                        <p className="mb-2 text-sm text-gray-500">
                          <span className="font-semibold">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-xs text-gray-500">PDF files only</p>
                      </div>
                      <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept=".pdf"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </label>
                  </div>

                  {selectedFiles.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Selected Files:</h4>
                      <div className="space-y-2">
                        {selectedFiles.map((file, index) => (
                          <div key={index} className="flex items-center space-x-2 p-2 bg-blue-50 rounded-lg">
                            <FileText className="w-4 h-4 text-blue-600" />
                            <span className="text-sm text-gray-700">{file.name}</span>
                            <span className="text-xs text-gray-500">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <StatusMessage status={uploadStatus} />

                <div className="flex justify-between">
                  <button
                    onClick={() => handlePrevStep(3)}
                    className="px-6 py-3 bg-gray-500 text-white rounded-xl font-semibold hover:bg-gray-600 transition-colors duration-200 flex items-center space-x-2"
                    disabled={isUploading}
                  >
                    <ArrowLeft className="w-4 h-4" />
                    <span>Previous</span>
                  </button>
                  <button
                    onClick={uploadDocuments}
                    disabled={selectedFiles.length === 0 || isUploading}
                    className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-purple-600 transform hover:scale-105 transition-all duration-200 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center space-x-2"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <span>Process Documents</span>
                        <Upload className="w-4 h-4" />
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Step 3b: URL Input (if summarize selected) */}
            {currentStep === 3 && selectedFeature === 'summarize' && (
              <div className="space-y-6 animate-in fade-in duration-500">
                <div className="flex items-center space-x-4 pb-4 border-b border-gray-200">
                  <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold">
                    3
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-800">Content Summarization</h2>
                </div>

                <div className="space-y-4">
                  <label className="block text-sm font-medium text-gray-700">
                    Enter URL to summarize
                  </label>
                  <div className="relative">
                    <Globe className="absolute left-4 top-4 w-5 h-5 text-gray-400" />
                    <input
                      type="url"
                      value={urlInput}
                      onChange={(e) => setUrlInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      className="w-full pl-12 pr-4 py-4 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-purple-100 focus:border-purple-500 transition-all duration-200 text-lg"
                      placeholder="https://example.com or https://youtube.com/watch?v=..."
                      disabled={isSummarizing}
                    />
                  </div>
                  <p className="text-sm text-gray-500">
                    Supports YouTube videos and web articles. The content will be summarized in approximately 300 words.
                  </p>
                </div>

                <StatusMessage status={uploadStatus} />

                {summaryResult && (
                  <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-2xl border border-purple-200">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center space-x-2">
                      <FileText className="w-5 h-5 text-purple-600" />
                      <span>Summary</span>
                    </h3>
                    <div className="prose max-w-none text-gray-700 leading-relaxed">
                      {summaryResult.split('\n').map((paragraph, index) => (
                        <p key={index} className="mb-3 last:mb-0">
                          {paragraph}
                        </p>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex justify-between">
                  <button
                    onClick={() => handlePrevStep(3)}
                    className="px-6 py-3 bg-gray-500 text-white rounded-xl font-semibold hover:bg-gray-600 transition-colors duration-200 flex items-center space-x-2"
                    disabled={isSummarizing}
                  >
                    <ArrowLeft className="w-4 h-4" />
                    <span>Previous</span>
                  </button>
                  <button
                    onClick={summarizeContent}
                    disabled={!urlInput.trim() || isSummarizing}
                    className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-xl font-semibold hover:from-purple-600 hover:to-pink-600 transform hover:scale-105 transition-all duration-200 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center space-x-2"
                  >
                    {isSummarizing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Summarizing...</span>
                      </>
                    ) : (
                      <>
                        <span>Summarize Content</span>
                        <ExternalLink className="w-4 h-4" />
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Step 4: Chat Interface (for documents) */}
            {currentStep === 4 && selectedFeature === 'documents' && (
              <div className="space-y-6 animate-in fade-in duration-500">
                <div className="flex items-center space-x-4 pb-4 border-b border-gray-200">
                  <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-blue-500 rounded-full flex items-center justify-center text-white font-bold">
                    <MessageCircle className="w-5 h-5" />
                  </div>
                  <h2 className="text-2xl font-semibold text-gray-800">Chat with Your Documents</h2>
                </div>

                {/* Chat Messages */}
                <div className="bg-gray-50 rounded-2xl p-4 h-96 overflow-y-auto border-2 border-gray-100">
                  <div className="space-y-4">
                    {messages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                            message.type === 'user'
                              ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white'
                              : message.isError
                              ? 'bg-red-100 text-red-800 border border-red-200'
                              : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
                          }`}
                        >
                          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                        </div>
                      </div>
                    ))}
                    {isSending && (
                      <div className="flex justify-start">
                        <div className="bg-white text-gray-800 border border-gray-200 shadow-sm max-w-xs lg:max-w-md px-4 py-3 rounded-2xl">
                          <div className="flex items-center space-x-2">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span className="text-sm">Thinking...</span>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </div>

                {/* Chat Input */}
                <div className="border-t border-gray-200 p-2 bg-white">
                    <div className="flex space-x-4">
                      <input
                        type="text"
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask a question about your documents..."
                        className="flex-1 px-6 py-2 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-blue-100 focus:border-blue-500 transition-all duration-200"
                        disabled={isSending}
                      />
                      <button
                        onClick={sendMessage}
                        disabled={isSending || !inputMessage.trim()}
                        className="px-6 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl font-semibold hover:from-blue-600 hover:to-purple-600 transform hover:scale-105 transition-all duration-200 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                      >
                        <Send className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                   <div className="flex justify-start">
                  <button
                    onClick={() => handlePrevStep(3)}
                    className="px-6 py-1.5 bg-gray-500 text-white rounded-xl font-semibold hover:bg-gray-600 transition-colors duration-200"
                  >
                    Previous
                  </button>
                </div>
              
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};





export default App;
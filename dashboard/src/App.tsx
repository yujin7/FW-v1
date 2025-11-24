import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { AlertTriangle, CheckCircle, XCircle, Activity, TrendingUp, TrendingDown, Clock, Zap, Database, RefreshCw, Bell, Settings, Eye, Shield } from 'lucide-react';

// Sample data generators
const generateTimeSeriesData = (days = 30) => {
  const data = [];
  let mae = 0.76;
  let auc = 0.93;

  for (let i = 0; i < days; i++) {
    mae += (Math.random() - 0.5) * 0.02;
    auc += (Math.random() - 0.5) * 0.01;
    data.push({
      date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      mae: Math.max(0.5, Math.min(1.2, mae)),
      auc: Math.max(0.85, Math.min(0.98, auc)),
      sharpe: 2.5 + (Math.random() - 0.5) * 0.4,
    });
  }
  return data;
};

const generatePredictions = (n = 10) => {
  const predictions = [];
  for (let i = 0; i < n; i++) {
    const date = new Date(Date.now() + i * 24 * 60 * 60 * 1000);
    predictions.push({
      date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      probability: Math.random() * 0.4,
      severity: -5 - Math.random() * 15,
      regime: ['Normal', 'Stress', 'Low Vol'][Math.floor(Math.random() * 3)],
    });
  }
  return predictions;
};

// Components
const MetricCard = ({ title, value, unit, trend, status, icon: Icon }) => {
  const statusColors = {
    good: 'bg-green-500',
    warning: 'bg-yellow-500',
    critical: 'bg-red-500',
  };

  const trendColors = {
    up: 'text-green-400',
    down: 'text-red-400',
    stable: 'text-gray-400',
  };

  return (
    <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm">{title}</span>
        <div className={`w-2 h-2 rounded-full ${statusColors[status]}`} />
      </div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-bold text-white">{value}</span>
        <span className="text-gray-500 text-sm mb-1">{unit}</span>
      </div>
      <div className={`flex items-center gap-1 mt-2 ${trendColors[trend]}`}>
        {trend === 'up' ? <TrendingUp size={14} /> : trend === 'down' ? <TrendingDown size={14} /> : <Activity size={14} />}
        <span className="text-xs">{trend === 'up' ? '+2.3%' : trend === 'down' ? '-1.5%' : 'Stable'}</span>
      </div>
    </div>
  );
};

const AlertItem = ({ type, message, time }) => {
  const icons = {
    critical: <XCircle className="text-red-400" size={18} />,
    warning: <AlertTriangle className="text-yellow-400" size={18} />,
    info: <CheckCircle className="text-blue-400" size={18} />,
  };

  const bgColors = {
    critical: 'bg-red-900/20 border-red-800',
    warning: 'bg-yellow-900/20 border-yellow-800',
    info: 'bg-blue-900/20 border-blue-800',
  };

  return (
    <div className={`p-3 rounded-lg border ${bgColors[type]} flex items-start gap-3`}>
      {icons[type]}
      <div className="flex-1">
        <p className="text-sm text-white">{message}</p>
        <p className="text-xs text-gray-500 mt-1">{time}</p>
      </div>
    </div>
  );
};

const TestStatusBadge = ({ passed, total }) => {
  const percentage = (passed / total) * 100;
  const color = percentage === 100 ? 'bg-green-500' : percentage >= 80 ? 'bg-yellow-500' : 'bg-red-500';

  return (
    <div className="flex items-center gap-2">
      <div className={`w-3 h-3 rounded-full ${color}`} />
      <span className="text-white font-medium">{passed}/{total}</span>
    </div>
  );
};

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState('30d');
  const [performanceData, setPerformanceData] = useState(generateTimeSeriesData(30));
  const [predictions, setPredictions] = useState(generatePredictions(10));

  const modelWeights = [
    { name: 'Ridge', weight: 0.35 },
    { name: 'ElasticNet', weight: 0.25 },
    { name: 'RF', weight: 0.20 },
    { name: 'GBM', weight: 0.15 },
    { name: 'GP', weight: 0.05 },
  ];

  const factorWeights = [
    { name: 'Western Aspects', weight: 0.28 },
    { name: 'Ashtakavarga', weight: 0.24 },
    { name: 'Fibonacci Price', weight: 0.18 },
    { name: 'Uranian', weight: 0.16 },
    { name: 'Bradley', weight: 0.14 },
  ];

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Eye },
    { id: 'performance', label: 'Performance', icon: Activity },
    { id: 'predictions', label: 'Predictions', icon: TrendingUp },
    { id: 'testing', label: 'Testing', icon: Shield },
    { id: 'learning', label: 'Learning', icon: RefreshCw },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Crash Prediction System</h1>
          <p className="text-gray-400">Model v34.0 - Self-Learning Active</p>
        </div>
        <div className="flex items-center gap-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
          <button className="p-2 bg-gray-800 rounded-lg hover:bg-gray-700">
            <Bell size={20} />
          </button>
          <button className="p-2 bg-gray-800 rounded-lg hover:bg-gray-700">
            <Settings size={20} />
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex gap-2 mb-6 border-b border-gray-800 pb-4">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
              activeTab === tab.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <tab.icon size={18} />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard title="Timing AUC" value="0.93" unit="" trend="up" status="good" />
            <MetricCard title="Severity MAE" value="0.76" unit="%" trend="stable" status="good" />
            <MetricCard title="Sharpe Ratio" value="2.7" unit="" trend="up" status="good" />
            <MetricCard title="Economic Value" value="$2.6M" unit="/yr" trend="up" status="good" />
          </div>

          {/* Main Chart */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h3 className="text-lg font-semibold mb-4">Performance Trend</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                />
                <Legend />
                <Area type="monotone" dataKey="auc" name="Timing AUC" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
                <Area type="monotone" dataKey="mae" name="Severity MAE" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Two Column Layout */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Recent Alerts */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Recent Alerts</h3>
              <div className="space-y-3">
                <AlertItem type="info" message="Model weights updated successfully" time="2 hours ago" />
                <AlertItem type="warning" message="Slight drift detected in factor weights" time="5 hours ago" />
                <AlertItem type="info" message="Weekly scheduled update complete" time="1 day ago" />
                <AlertItem type="critical" message="High crash probability detected: Oct 19, 2026" time="3 days ago" />
              </div>
            </div>

            {/* System Status */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4">System Status</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Data Pipeline</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-500" />
                    <span className="text-green-400">Healthy</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Model Serving</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-500" />
                    <span className="text-green-400">Active</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Drift Detection</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-yellow-500" />
                    <span className="text-yellow-400">Monitoring</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Current Regime</span>
                  <span className="text-blue-400 font-medium">Normal</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Last Learning Event</span>
                  <span className="text-gray-300">2 hours ago</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-8 pt-4 border-t border-gray-800 text-center text-gray-500 text-sm">
        <p>Crash Prediction System v34.0 - Self-Learning Engine Active - Last sync: {new Date().toLocaleTimeString()}</p>
      </div>
    </div>
  );
};

export default Dashboard;

import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { ShieldAlert, AlertTriangle, Clock, Search, RefreshCw, Zap, Radar } from 'lucide-react';

const ALERT_COLORS = {
  high: 'bg-red-100 text-red-800 border-red-200',
  medium: 'bg-amber-100 text-amber-800 border-amber-200',
  low: 'bg-blue-100 text-blue-800 border-blue-200',
};

const PIE_COLORS = ['#ef4444', '#f59e0b', '#3b82f6'];

function severityWeight(severity) {
  if (severity === 'high') return 3;
  if (severity === 'medium') return 2;
  return 1;
}

function riskLevel(score) {
  if (score >= 70) return 'Critical';
  if (score >= 45) return 'Elevated';
  if (score >= 20) return 'Guarded';
  return 'Stable';
}

export default function ThreatDashboard() {
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [trafficOverTime, setTrafficOverTime] = useState([]);
  const [stats, setStats] = useState({
    alerts_count: 0,
    active_hosts: 0,
  });
  const [filterValue, setFilterValue] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    fetchDashboardData();
    window.addEventListener('dashboardUpdate', handleWebSocketUpdate);
    return () => {
      window.removeEventListener('dashboardUpdate', handleWebSocketUpdate);
    };
  }, []);

  const fetchDashboardData = async () => {
    setIsLoading(true);
    try {
      const [alertsRes, trendRes, statsRes] = await Promise.all([
        fetch('/api/recent-alerts/'),
        fetch('/api/traffic-over-time/'),
        fetch('/api/dashboard-stats/'),
      ]);
      const [alertsData, trendData, statsData] = await Promise.all([
        alertsRes.json(),
        trendRes.json(),
        statsRes.json(),
      ]);
      setRecentAlerts(Array.isArray(alertsData) ? alertsData : []);
      setTrafficOverTime(Array.isArray(trendData) ? trendData : []);
      setStats(statsData || {});
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleWebSocketUpdate = (event) => {
    const update = event.detail;
    if (update?.type === 'alert_update' && update?.data) {
      setRecentAlerts((prev) => [update.data, ...prev].slice(0, 50));
      setStats((prev) => ({
        ...prev,
        alerts_count: (prev.alerts_count || 0) + 1,
      }));
      setLastUpdate(new Date());
    }
  };

  const filteredAlerts = useMemo(() => {
    if (!filterValue) return recentAlerts;
    const q = filterValue.toLowerCase();
    return recentAlerts.filter((alert) => {
      const hay = [
        alert.src_ip,
        alert.dst_ip,
        alert.protocol,
        alert.label,
        alert.threat_type,
        alert.threat_family,
        alert.threat_detail,
        alert.detection_source,
        alert.confidence != null ? String(alert.confidence) : '',
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return hay.includes(q);
    });
  }, [recentAlerts, filterValue]);

  const severityCounts = useMemo(() => {
    const counts = { high: 0, medium: 0, low: 0 };
    recentAlerts.forEach((a) => {
      const s = (a.severity || 'low').toLowerCase();
      if (counts[s] !== undefined) counts[s] += 1;
    });
    return counts;
  }, [recentAlerts]);

  const severityPie = useMemo(
    () => [
      { name: 'High', value: severityCounts.high },
      { name: 'Medium', value: severityCounts.medium },
      { name: 'Low', value: severityCounts.low },
    ].filter((x) => x.value > 0),
    [severityCounts]
  );

  const familyBreakdown = useMemo(() => {
    const counts = {};
    recentAlerts.forEach((a) => {
      const family = a.threat_family || 'Unknown';
      counts[family] = (counts[family] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }, [recentAlerts]);

  const topThreatTypes = useMemo(() => {
    const counts = {};
    recentAlerts.forEach((a) => {
      const t = a.threat_type || 'UNCLASSIFIED';
      counts[t] = (counts[t] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
  }, [recentAlerts]);

  const threatTrend = useMemo(
    () =>
      (trafficOverTime || []).map((slot) => ({
        hour: slot.hour,
        alerts: slot.suspicious || 0,
      })),
    [trafficOverTime]
  );

  const riskScore = useMemo(() => {
    const totalWeighted =
      severityCounts.high * 3 + severityCounts.medium * 2 + severityCounts.low;
    return Math.min(100, totalWeighted * 4);
  }, [severityCounts]);

  const formattedLastUpdate = lastUpdate.toLocaleTimeString();
  const risk = riskLevel(riskScore);

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-slate-900 text-white shadow-lg">
        <div className="container mx-auto px-4 py-4 flex flex-col md:flex-row gap-3 md:gap-0 md:justify-between md:items-center">
          <div className="flex items-center gap-2">
            <ShieldAlert size={28} />
            <div>
              <h1 className="text-2xl font-bold">Threat Command View</h1>
              <p className="text-sm text-slate-300">Big-picture intrusion signals only</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-sm bg-slate-800 rounded-md px-3 py-1.5 flex items-center gap-2">
              <Clock size={14} />
              <span>Last update: {formattedLastUpdate}</span>
            </div>
            <button
              onClick={fetchDashboardData}
              className="bg-white text-slate-900 px-4 py-2 rounded-md font-medium hover:bg-slate-100 flex items-center"
              disabled={isLoading}
            >
              <RefreshCw size={16} className={`mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </header>

      {isLoading && (
        <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg flex items-center">
            <RefreshCw size={22} className="animate-spin mr-3 text-slate-700" />
            <span className="font-medium">Loading threat intelligence...</span>
          </div>
        </div>
      )}

      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-5 border border-slate-100">
            <p className="text-xs uppercase text-slate-500">Current Risk</p>
            <p className="text-3xl font-bold text-slate-900 mt-1">{riskScore}</p>
            <p className="text-sm text-slate-600">{risk} posture</p>
          </div>
          <div className="bg-white rounded-lg shadow p-5 border border-slate-100">
            <p className="text-xs uppercase text-slate-500">Total Alerts</p>
            <p className="text-3xl font-bold text-slate-900 mt-1">{(stats.alerts_count || 0).toLocaleString()}</p>
            <p className="text-sm text-slate-600">In active window</p>
          </div>
          <div className="bg-white rounded-lg shadow p-5 border border-slate-100">
            <p className="text-xs uppercase text-slate-500">High Severity</p>
            <p className="text-3xl font-bold text-red-600 mt-1">{severityCounts.high}</p>
            <p className="text-sm text-slate-600">Immediate triage</p>
          </div>
          <div className="bg-white rounded-lg shadow p-5 border border-slate-100">
            <p className="text-xs uppercase text-slate-500">Affected Hosts</p>
            <p className="text-3xl font-bold text-slate-900 mt-1">{(stats.active_hosts || 0).toLocaleString()}</p>
            <p className="text-sm text-slate-600">Potential blast radius</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="bg-white rounded-lg shadow p-5 lg:col-span-2">
            <div className="flex items-center gap-2 mb-4">
              <Radar size={18} className="text-slate-700" />
              <h2 className="text-lg font-semibold">Threat Trend (suspicious events)</h2>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={threatTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="alerts" stroke="#ef4444" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="bg-white rounded-lg shadow p-5">
            <h2 className="text-lg font-semibold mb-4">Severity Mix</h2>
            {severityPie.length === 0 ? (
              <div className="h-[280px] flex items-center justify-center text-slate-500">No threats yet</div>
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={severityPie}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={90}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {severityPie.map((entry, index) => (
                      <Cell key={`severity-cell-${entry.name}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <div className="bg-white rounded-lg shadow p-5">
            <h2 className="text-lg font-semibold mb-4">Threat Families</h2>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={familyBreakdown}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" hide />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" />
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-3 text-xs text-slate-600 space-y-1">
              {familyBreakdown.map((item) => (
                <div key={`family-label-${item.name}`} className="flex justify-between">
                  <span className="truncate mr-2">{item.name}</span>
                  <span>{item.count}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="bg-white rounded-lg shadow p-5 lg:col-span-2">
            <h2 className="text-lg font-semibold mb-3">Top Threat Types</h2>
            <div className="grid md:grid-cols-2 gap-3">
              {topThreatTypes.length > 0 ? (
                topThreatTypes.map((t) => (
                  <div key={t.name} className="border border-slate-200 rounded-lg p-3 bg-slate-50">
                    <p className="text-xs text-slate-500">Threat Type</p>
                    <p className="font-mono text-sm font-semibold text-slate-800 break-all">{t.name}</p>
                    <p className="text-sm mt-1 text-slate-600">Seen {t.count} times</p>
                  </div>
                ))
              ) : (
                <div className="text-slate-500">No classified threats available.</div>
              )}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-5">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
            <h2 className="text-lg font-semibold">Live Incident Feed</h2>
            <div className="relative">
              <Search size={16} className="absolute left-3 top-2.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search type, family, source, destination..."
                className="w-full md:w-96 pl-9 pr-4 py-2 bg-slate-100 rounded-md focus:outline-none focus:ring-2 focus:ring-slate-400"
                value={filterValue}
                onChange={(e) => setFilterValue(e.target.value)}
              />
            </div>
          </div>

          <div className="overflow-y-auto max-h-[32rem] pr-1">
            {filteredAlerts.length > 0 ? (
              filteredAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`mb-3 p-3 border rounded-lg ${ALERT_COLORS[alert.severity] || 'bg-slate-100 text-slate-800 border-slate-200'}`}
                >
                  <div className="flex justify-between gap-2">
                    <div className="flex items-center min-w-0">
                      {alert.severity === 'high' && <Zap size={15} className="mr-1 text-red-600 shrink-0" />}
                      <span className="font-medium truncate">{alert.label || 'Suspicious activity'}</span>
                    </div>
                    <span className="text-xs md:text-sm shrink-0">{alert.timestamp}</span>
                  </div>

                  <div className="text-xs mt-1 font-mono text-slate-700 flex flex-wrap gap-x-2 gap-y-1">
                    {alert.threat_type && (
                      <span className="bg-white/60 px-1.5 rounded border border-slate-200">{alert.threat_type}</span>
                    )}
                    {alert.threat_family && <span>{alert.threat_family}</span>}
                    {alert.detection_source && <span>{alert.detection_source}</span>}
                    {alert.confidence != null && <span>{`${Math.round(Number(alert.confidence) * 100)}%`}</span>}
                    {alert.severity && <span className="uppercase">{alert.severity}</span>}
                  </div>

                  <div className="text-sm mt-1">
                    {alert.src_ip || 'Unknown'} → {alert.dst_ip || 'Unknown'} ({alert.protocol || 'N/A'})
                  </div>

                  {alert.threat_detail && (
                    <pre className="text-xs mt-2 whitespace-pre-wrap text-slate-600 max-h-28 overflow-y-auto font-sans leading-snug">
                      {alert.threat_detail}
                    </pre>
                  )}
                </div>
              ))
            ) : (
              <div className="text-center py-10 text-slate-500 flex items-center justify-center gap-2">
                <AlertTriangle size={18} />
                <span>No incidents matching this filter</span>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

const rootElement = document.getElementById('dashboard-root');
if (rootElement) {
  const root = createRoot(rootElement);
  root.render(<ThreatDashboard />);
}
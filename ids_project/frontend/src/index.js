import { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import { Shield, AlertTriangle, Activity, Clock, Server, Database, Search, RefreshCw, Zap } from 'lucide-react';

// Constants
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];
const ALERT_COLORS = {
  high: 'bg-red-100 text-red-800 border-red-200',
  medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  low: 'bg-blue-100 text-blue-800 border-blue-200',
};

export default function NetworkDashboard() {
  // State for dashboard data
  const [trafficOverTime, setTrafficOverTime] = useState([]);
  const [protocolDistribution, setProtocolDistribution] = useState([]);
  const [topSources, setTopSources] = useState([]);
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [stats, setStats] = useState({
    total_traffic: 0,
    alerts_count: 0,
    active_hosts: 0,
    data_transferred: 0
  });
  
  // UI state
  const [filterValue, setFilterValue] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('24h');
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Fetch initial data when component mounts
  useEffect(() => {
    fetchDashboardData();
    
    // Listen for WebSocket updates
    window.addEventListener('dashboardUpdate', handleWebSocketUpdate);
    
    // Cleanup
    return () => {
      window.removeEventListener('dashboardUpdate', handleWebSocketUpdate);
    };
  }, []);

  // Fetch all dashboard data from the API
  const fetchDashboardData = async () => {
    setIsLoading(true);
    try {
      // Fetch traffic over time
      const trafficResponse = await fetch('/api/traffic-over-time/');
      const trafficData = await trafficResponse.json();
      setTrafficOverTime(trafficData);
      
      // Fetch protocol distribution
      const protocolResponse = await fetch('/api/protocol-distribution/');
      const protocolData = await protocolResponse.json();
      setProtocolDistribution(protocolData);
      
      // Fetch top sources
      const sourcesResponse = await fetch('/api/top-sources/');
      const sourcesData = await sourcesResponse.json();
      setTopSources(sourcesData);
      
      // Fetch recent alerts
      const alertsResponse = await fetch('/api/recent-alerts/');
      const alertsData = await alertsResponse.json();
      setRecentAlerts(alertsData);
      
      // Fetch dashboard stats
      const statsResponse = await fetch('/api/dashboard-stats/');
      const statsData = await statsResponse.json();
      setStats(statsData);
      
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle WebSocket updates
  const handleWebSocketUpdate = (event) => {
    const update = event.detail;
    
    if (update.type === 'traffic_update') {
      // Update traffic stats
      setStats(prev => ({
        ...prev,
        total_traffic: prev.total_traffic + 1
      }));
      
      // Could also update traffic over time chart here
    } 
    else if (update.type === 'alert_update') {
      // Add new alert to recent alerts
      setRecentAlerts(prev => {
        const newAlerts = [update.data, ...prev].slice(0, 20);
        return newAlerts;
      });
      
      // Update alert count
      setStats(prev => ({
        ...prev,
        alerts_count: prev.alerts_count + 1
      }));
    }
    
    setLastUpdate(new Date());
  };

  // Filter alerts based on search input
  const filteredAlerts = recentAlerts.filter(alert => 
    (alert.src_ip && alert.src_ip.includes(filterValue)) || 
    (alert.dst_ip && alert.dst_ip.includes(filterValue)) || 
    (alert.protocol && alert.protocol.includes(filterValue)) || 
    (alert.label && alert.label.includes(filterValue))
  );
  
  // Format the last update time
  const formattedLastUpdate = lastUpdate.toLocaleTimeString();

  return (
    <div className="bg-gray-50 min-h-screen">
      {/* Header */}
      <header className="bg-indigo-600 text-white shadow-lg">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Shield size={28} />
            <h1 className="text-2xl font-bold">Network IDS Dashboard</h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-sm bg-indigo-700 rounded-md px-3 py-1">
              <span className="mr-2">Last update:</span>
              <span>{formattedLastUpdate}</span>
            </div>
            <button 
              onClick={fetchDashboardData}
              className="bg-white text-indigo-600 px-4 py-2 rounded-md font-medium hover:bg-indigo-50 flex items-center"
              disabled={isLoading}
            >
              {isLoading ? (
                <RefreshCw size={18} className="mr-2 animate-spin" />
              ) : (
                <Activity size={18} className="mr-2" />
              )}
              {isLoading ? 'Refreshing...' : 'Refresh Data'}
            </button>
            <div className="relative">
              <Clock size={20} className="absolute left-3 top-2.5 text-indigo-300" />
              <select 
                className="pl-10 pr-4 py-2 bg-indigo-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-white"
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
              >
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
            </div>
          </div>
        </div>
      </header>

      {/* Loading overlay */}
      {isLoading && (
        <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg flex items-center">
            <RefreshCw size={24} className="animate-spin mr-3 text-indigo-600" />
            <span className="text-lg font-medium">Loading dashboard data...</span>
          </div>
        </div>
      )}

      {/* Dashboard Content */}
      <main className="container mx-auto px-4 py-6">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="bg-white rounded-lg shadow p-6 flex items-center">
            <div className="rounded-full bg-blue-100 p-3 mr-4">
              <Activity size={24} className="text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Traffic</p>
              <p className="text-2xl font-bold">{stats.total_traffic.toLocaleString()} packets</p>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6 flex items-center">
            <div className="rounded-full bg-red-100 p-3 mr-4">
              <AlertTriangle size={24} className="text-red-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Alerts</p>
              <p className="text-2xl font-bold">{stats.alerts_count.toLocaleString()} detected</p>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6 flex items-center">
            <div className="rounded-full bg-green-100 p-3 mr-4">
              <Server size={24} className="text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Active Hosts</p>
              <p className="text-2xl font-bold">{stats.active_hosts.toLocaleString()} devices</p>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6 flex items-center">
            <div className="rounded-full bg-purple-100 p-3 mr-4">
              <Database size={24} className="text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Data Transferred</p>
              <p className="text-2xl font-bold">{stats.data_transferred.toLocaleString()} GB</p>
            </div>
          </div>
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Traffic Over Time */}
          <div className="bg-white rounded-lg shadow p-6 col-span-2">
            <h2 className="text-lg font-semibold mb-4">Traffic Over Time</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trafficOverTime}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Packet Count', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="normal" stroke="#8884d8" name="Normal Traffic" strokeWidth={2} />
                <Line type="monotone" dataKey="suspicious" stroke="#ff7300" name="Suspicious Traffic" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Protocol Distribution */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Protocol Distribution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={protocolDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {protocolDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Sources and Alerts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Source IPs */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Top Source IPs</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={topSources}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="src_ip" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8884d8">
                  {topSources.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Recent Alerts */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold">Recent Alerts</h2>
              <div className="relative">
                <Search size={16} className="absolute left-3 top-2.5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Filter alerts..."
                  className="pl-9 pr-4 py-2 bg-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  value={filterValue}
                  onChange={(e) => setFilterValue(e.target.value)}
                />
              </div>
            </div>
            <div className="overflow-y-auto max-h-64">
              {filteredAlerts.length > 0 ? (
                filteredAlerts.map(alert => (
                  <div 
                    key={alert.id} 
                    className={`mb-3 p-3 border rounded-lg ${ALERT_COLORS[alert.severity] || 'bg-gray-100'}`}
                  >
                    <div className="flex justify-between">
                      <div className="flex items-center">
                        {alert.severity === 'high' && <Zap size={16} className="mr-1 text-red-600" />}
                        <span className="font-medium">{alert.label}</span>
                      </div>
                      <span className="text-sm">{alert.timestamp}</span>
                    </div>
                    <div className="text-sm mt-1">
                      {alert.src_ip} → {alert.dst_ip} ({alert.protocol})
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-6 text-gray-500">No alerts matching your filter</div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
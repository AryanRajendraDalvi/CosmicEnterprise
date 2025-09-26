# Cosmic Weather Insurance App

A comprehensive space weather risk assessment and insurance pricing system with a React frontend and Python backend.

## Project Structure

```
├── Backend/
│   ├── Cosmicpls.py         # Main Python backend system
│   ├── api.py              # Flask API to expose backend functionality
│   ├── requirements.txt    # Python dependencies
│   └── start_api.py        # Script to start the backend API
├── src/                    # React frontend source code
│   ├── components/         # React components
│   ├── services/           # API service layer
│   └── App.tsx             # Main application component
├── README.md               # This file
└── package.json            # Frontend dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

### Backend Setup

1. Navigate to the Backend directory:
   ```bash
   cd Backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend API:
   ```bash
   python api.py
   ```
   
   Or on Windows, you can run:
   ```bash
   start_api.bat
   ```

   The backend will start on http://localhost:5000

### Frontend Setup

1. Install frontend dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will start on http://localhost:3000

## API Endpoints

The backend API provides the following endpoints:

- `GET /api/health` - Health check
- `GET /api/satellites` - Get list of satellites with optional search
- `GET /api/satellites/operators` - Get list of satellite operators
- `GET /api/satellites/{satellite_name}` - Get detailed satellite information
- `POST /api/forecast` - Get space weather forecast
- `POST /api/risk-assessment` - Get risk assessment for a satellite
- `GET /api/database/stats` - Get satellite database statistics

## Features

- Real-time space weather monitoring and forecasting
- Comprehensive satellite database with 5000+ satellites
- Risk assessment based on orbital parameters, shielding, and age
- Insurance premium calculation with Monte Carlo simulation
- Interactive dashboard with mission control aesthetics
- Real-time data visualization
- **Data Visualization Dashboard** - Comprehensive charts and graphs for all system data
- **Satellite Details Panel** - Detailed information about selected satellites
- **Enhanced Risk Assessment Visualization** - Charts showing risk factors and financial impact

## Technologies Used

### Backend
- Python
- Flask (web framework)
- Pandas (data analysis)
- NumPy (numerical computing)
- Scikit-learn (machine learning)
- TensorFlow (neural networks)

### Frontend
- React
- TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- Recharts (data visualization)
- Lucide React (icons)

## New Data Visualization Features

### Data Visualization Dashboard
The new dashboard provides comprehensive visualizations of all system data:
- Satellite orbit distribution (pie chart)
- Satellite type distribution (bar chart)
- Top 10 most valuable satellites (horizontal bar chart)
- Top 10 heaviest satellites (horizontal bar chart)
- Satellite launch timeline (area chart)
- Operator distribution (bar chart)
- Space weather forecast (line chart)

### Satellite Details Panel
Detailed information view for selected satellites:
- Technical specifications
- Mission information
- Risk assessment factors with visual indicators
- Key metrics display

### Enhanced Risk Assessment
Visual representation of risk factors:
- Bar charts showing risk factor analysis
- Pie charts for risk factor distribution
- Financial impact visualization

## Development

### Backend Development

The main backend logic is in `Cosmicpls.py` which contains:
- `SpaceWeatherDataIngester` - Data ingestion and synthesis
- `SatelliteDatabase` - Satellite data management
- `SpaceWeatherForecaster` - ML-based forecasting
- `RiskImpactModeler` - Risk modeling
- `InsurancePricer` - Premium calculation

The Flask API in `api.py` exposes these functionalities as REST endpoints.

### Frontend Development

The React frontend is structured with:
- `App.tsx` - Main application component with navigation
- `components/` - Individual UI components
- `services/api.ts` - API service layer
- `components/DataVisualizationDashboard.tsx` - New comprehensive dashboard
- `components/SatelliteDetailsPanel.tsx` - Detailed satellite information
- Enhanced visualization in `RiskAssessment.tsx` and `SatelliteSelector.tsx`

All components are designed with a "mission control" aesthetic using Tailwind CSS.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License.
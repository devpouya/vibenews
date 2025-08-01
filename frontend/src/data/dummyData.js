// Dummy data for VibeNews frontend demo

export const topics = [
  {
    id: 'russia-ukraine-conflict',
    title: 'Russia-Ukraine Conflict',
    description: 'Analysis of media coverage regarding the ongoing conflict between Russia and Ukraine',
    articleCount: 156,
    lastUpdated: '2025-08-01T10:30:00Z',
    icon: '🇺🇦',
    gradient: 'from-blue-500 to-yellow-400'
  },
  {
    id: 'climate-change-policy',
    title: 'Climate Change Policy',
    description: 'Coverage of climate policies, environmental regulations, and green energy initiatives',
    articleCount: 234,
    lastUpdated: '2025-08-01T09:15:00Z',
    icon: '🌍',
    gradient: 'from-green-500 to-emerald-400'
  },
  {
    id: 'economic-inflation',
    title: 'Economic Inflation',
    description: 'Analysis of inflation coverage, monetary policy, and economic impact reporting',
    articleCount: 189,
    lastUpdated: '2025-08-01T11:20:00Z',
    icon: '📈',
    gradient: 'from-purple-500 to-pink-400'
  },
  {
    id: 'immigration-policy',
    title: 'Immigration Policy',
    description: 'Coverage of immigration laws, border policies, and refugee situations',
    articleCount: 127,
    lastUpdated: '2025-08-01T08:45:00Z',
    icon: '🌐',
    gradient: 'from-orange-500 to-red-400'
  },
  {
    id: 'tech-regulation',
    title: 'Tech Regulation',
    description: 'Analysis of technology policy, AI regulation, and digital privacy laws',
    articleCount: 203,
    lastUpdated: '2025-08-01T12:00:00Z',
    icon: '💻',
    gradient: 'from-indigo-500 to-blue-400'
  }
];

export const articles = {
  'russia-ukraine-conflict': [
    {
      id: 1,
      title: "Ukraine's Counteroffensive Shows Significant Progress in Eastern Front",
      source: "Swiss News Today",
      publishedDate: "2025-08-01T10:30:00Z",
      biasScore: -0.7,
      biasLabel: "Strongly Pro-Ukraine",
      confidence: 0.89,
      url: "#",
      aiSummary: "This article emphasizes Ukrainian military successes and uses positive language when describing Ukrainian operations, while minimizing Russian strategic positions. The headline frames Ukraine as actively succeeding rather than neutrally describing military developments.",
      aiReasoning: "Language analysis reveals 78% positive sentiment toward Ukrainian forces, frequent use of success-oriented terminology ('significant progress', 'breakthrough'), and limited coverage of Russian perspectives. Sources quoted are predominantly Ukrainian military officials."
    },
    {
      id: 2,
      title: "Diplomatic Efforts Continue as Both Sides Assess Military Positions",
      source: "Neutral Observer",
      publishedDate: "2025-08-01T09:15:00Z",
      biasScore: 0.1,
      biasLabel: "Neutral",
      confidence: 0.95,
      url: "#",
      aiSummary: "Balanced reporting that presents both Ukrainian and Russian diplomatic positions without favoring either side. Uses neutral language and quotes multiple sources from different perspectives.",
      aiReasoning: "Analysis shows 52% neutral language, equal representation of both sides' viewpoints, and careful use of qualified statements. Sources include officials from both countries as well as international observers."
    },
    {
      id: 3,
      title: "Russian Defense Ministry Reports Strategic Repositioning in Key Sectors",
      source: "Eastern European Times",
      publishedDate: "2025-08-01T08:45:00Z",
      biasScore: 0.6,
      biasLabel: "Moderately Pro-Russia",
      confidence: 0.82,
      url: "#",
      aiSummary: "This article predominantly features Russian military statements and frames Russian troop movements as strategic rather than defensive. Ukrainian perspectives are limited or portrayed skeptically.",
      aiReasoning: "Content analysis shows 68% of quotes from Russian sources, strategic framing of Russian actions ('repositioning' vs 'retreat'), and questioning tone toward Ukrainian claims. Limited fact-checking of Russian statements."
    },
    {
      id: 4,
      title: "International Aid Package Reaches Ukraine Amid Ongoing Humanitarian Crisis",
      source: "Swiss Humanitarian Review",
      publishedDate: "2025-08-01T07:30:00Z",
      biasScore: -0.4,
      biasLabel: "Moderately Pro-Ukraine",
      confidence: 0.76,
      url: "#",
      aiSummary: "Focuses on humanitarian impact on Ukrainian civilians and effectiveness of international aid, with limited discussion of broader conflict context or Russian civilian impacts.",
      aiReasoning: "Humanitarian framing emphasizes Ukrainian suffering (82% of humanitarian examples), positive portrayal of international aid efforts, and minimal coverage of humanitarian issues in Russian-controlled territories."
    },
    {
      id: 5,
      title: "Energy Infrastructure Attacks Raise Concerns About Winter Preparations",
      source: "Energy Market Analyst",
      publishedDate: "2025-08-01T11:20:00Z",
      biasScore: -0.2,
      biasLabel: "Slightly Pro-Ukraine",
      confidence: 0.71,
      url: "#",
      aiSummary: "Technical analysis of infrastructure damage with slight emphasis on Ukrainian resilience and preparation efforts. Presents energy security as primarily a Ukrainian challenge.",
      aiReasoning: "Infrastructure analysis focuses 65% on Ukrainian facilities, emphasizes Ukrainian adaptation and resilience, with limited discussion of energy impacts on Russian economy or infrastructure."
    },
    {
      id: 6,
      title: "Moscow Announces New Military Doctrine Amid Regional Tensions",
      source: "Geopolitical Digest",
      publishedDate: "2025-08-01T06:15:00Z",
      biasScore: 0.8,
      biasLabel: "Strongly Pro-Russia",
      confidence: 0.87,
      url: "#",
      aiSummary: "Presents Russian military doctrine as defensive response to NATO expansion, with limited critical analysis of Russian claims or Ukrainian security concerns.",
      aiReasoning: "Analysis shows uncritical acceptance of Russian framing (73% of content), defensive language for Russian actions, and portrayal of NATO/Ukraine as aggressors. Limited fact-checking of Russian claims about threats."
    }
  ],
  'climate-change-policy': [
    {
      id: 7,
      title: "Switzerland Announces Ambitious Carbon Neutrality Timeline",
      source: "Green Policy Weekly",
      publishedDate: "2025-08-01T10:00:00Z",
      biasScore: -0.6,
      biasLabel: "Strongly Pro-Environment",
      confidence: 0.91,
      url: "#",
      aiSummary: "Highly positive coverage of Switzerland's climate commitments with limited discussion of economic costs or implementation challenges.",
      aiReasoning: "Environmental framing dominates (84% positive environmental language), minimal coverage of business concerns, and optimistic projections without thorough cost-benefit analysis."
    },
    {
      id: 8,
      title: "Business Leaders Express Mixed Views on New Environmental Regulations",
      source: "Economic Balance Today",
      publishedDate: "2025-08-01T09:30:00Z",
      biasScore: 0.0,
      biasLabel: "Neutral",
      confidence: 0.93,
      url: "#",
      aiSummary: "Balanced reporting presenting both environmental benefits and economic concerns of new regulations, with equal representation of stakeholder viewpoints.",
      aiReasoning: "Analysis shows balanced source representation (48% business, 47% environmental, 5% government), neutral language throughout, and careful presentation of multiple perspectives."
    }
  ]
};

export const getBiasColor = (score) => {
  if (score <= -0.6) return 'bg-bias-strong-green text-white';
  if (score <= -0.3) return 'bg-bias-moderate-green text-white';
  if (score <= -0.1) return 'bg-bias-light-green text-white';
  if (score >= 0.6) return 'bg-bias-strong-red text-white';
  if (score >= 0.3) return 'bg-bias-moderate-red text-white';
  if (score >= 0.1) return 'bg-bias-light-red text-white';
  return 'bg-bias-neutral text-white';
};

export const getBiasLabel = (score) => {
  if (score <= -0.6) return 'Strongly Pro-Ukraine';
  if (score <= -0.3) return 'Moderately Pro-Ukraine';
  if (score <= -0.1) return 'Slightly Pro-Ukraine';
  if (score >= 0.6) return 'Strongly Pro-Russia';
  if (score >= 0.3) return 'Moderately Pro-Russia';
  if (score >= 0.1) return 'Slightly Pro-Russia';
  return 'Neutral';
};

export const sortOptions = [
  { value: 'bias-desc', label: 'Most Pro-Russia First' },
  { value: 'bias-asc', label: 'Most Pro-Ukraine First' },
  { value: 'date-desc', label: 'Newest First' },
  { value: 'date-asc', label: 'Oldest First' },
  { value: 'confidence-desc', label: 'Highest Confidence First' },
  { value: 'confidence-asc', label: 'Lowest Confidence First' },
  { value: 'source-asc', label: 'Source A-Z' },
  { value: 'source-desc', label: 'Source Z-A' }
];
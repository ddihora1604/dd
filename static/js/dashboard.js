let activeQuery = '';
let startDate = '';
let endDate = '';
let uploadedData = true;  
let analysisPerformed = false; 

function showLoading(show) {
    const loadingElement = document.getElementById('loading');
    if (show) {
        loadingElement.style.display = 'block';
    } else {
        loadingElement.style.display = 'none';
    }
}

// Function to get dynamic description for a section - disabled
async function getDynamicDescription(section, query, context = {}) {
    // This function has been disabled to prevent errors with AI-generated descriptions
    return "";
}

// Function to update section description - disabled
async function updateSectionDescription(sectionId, descriptionElementSelector, context = {}) {
    // This function has been disabled to prevent errors with AI-generated descriptions
    return;
}

// Analysis button click handler
document.getElementById('analyze-btn').addEventListener('click', async () => {
    const query = document.getElementById('query-input').value;
    if (!query) {
        alert('Please enter a search query');
        return;
    }

    activeQuery = query;
    startDate = '';
    endDate = '';
    
    // Map of section IDs to user-friendly section titles
    const sectionTitles = {
        'ai_insights': 'Social Media Insights',
        'data_story': 'Comprehensive Data Story',
        'word_cloud': 'Word Cloud',
        'contributors': 'Top Contributors',
        'metrics': 'Key Metrics',
        'timeseries': 'Time Series Analysis',
        'topic_evolution': 'Topic Evolution',
        'network': 'User Network',
        'topics': 'Topic Analysis',
        'semantic_map': 'Semantic Map',
        'coordinated': 'Coordinated Behavior',
        'coordinated_groups': 'Coordinated Content Groups',
        'community_distribution': 'Community Distribution'
    };
    
        // Description areas loading has been disabled
    // No descriptions to update
    
    // Update the semantic map container and topic clusters with loading indicators
    document.getElementById('semantic-map-container').innerHTML = `
        <div class="section-loading">
            <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
            Loading ${sectionTitles['semantic_map']}...
        </div>
    `;
    
    document.getElementById('topic-clusters').innerHTML = `
        <p class="text-muted"><span class="spinner-border spinner-border-sm text-primary" role="status"></span> Loading Topic Clusters...</p>
    `;
    
    showLoading(true);
    
    try {
        // Clear all previous visualizations first - with null checks
        const clearElement = (id, defaultContent = '') => {
            const element = document.getElementById(id);
            if (element) {
                element.innerHTML = defaultContent;
            }
        };
        
        clearElement('topics-container');
        clearElement('contributors-overview');
        // Remove reference to network-graph which is no longer needed
        clearElement('coordinated-groups');
        clearElement('word-cloud');
        clearElement('timeseries-chart');
        clearElement('ai-summary'); // This element might not exist
        clearElement('topic-evolution-chart');
        clearElement('semantic-map-container');
        clearElement('point-details', '<p class="text-muted">Click on a point to see details</p>');
        clearElement('topic-clusters', '<p class="text-muted">Loading topic clusters...</p>');
        clearElement('community-distribution');
        
        // Reset data story with placeholder
        clearElement('data-story', `
            <div class="card-body">
                <p class="text-muted">Enter a search query to generate a data story or case study.</p>
            </div>
        `);
        
        // PERFORMANCE OPTIMIZATION: Load data progressively in phases
        // Phase 1: First load critical data for the overview tab
        const criticalPromises = [
            // Update key metrics - essential for overview
            updateKeyMetrics(query).catch(error => {
                console.error('Error updating key metrics:', error);
                // Error state is handled within updateKeyMetrics
            }),
            // Update top contributors (small visualization) - essential for overview
            updateContributorsOverview(query).catch(error => {
                console.error('Error updating contributors overview:', error);
                clearElement('contributors-overview', '<p class="text-danger">Error loading contributors data</p>');
            }),
            // Update word cloud - lightweight visualization for overview
            updateWordCloud(query).catch(error => {
                console.error('Error updating word cloud:', error);
                clearElement('word-cloud', '<p class="text-danger">Error loading word cloud data</p>');
            }),
        ];

        // Wait for critical components first
        await Promise.allSettled(criticalPromises);
        
        // Description updates have been disabled
        // No descriptions to update
        
        // Hide the main loading spinner as critical content is loaded
        showLoading(false);
        
        // Mark that analysis has been performed
        analysisPerformed = true;
        
        // PERFORMANCE OPTIMIZATION: Create placeholder loading indicators for remaining components
        clearElement('timeseries-chart', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading time series analysis...</div>');
        clearElement('topic-evolution-chart', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading topic evolution analysis...</div>');
        // Remove reference to network-graph which no longer exists
        clearElement('topics-container', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading topic analysis...</div>');
        clearElement('coordinated-groups', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading coordinated behavior analysis...</div>');
        clearElement('community-distribution', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading community distribution...</div>');
        clearElement('semantic-map-container', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading semantic map analysis...</div>');
        
        // Phase 2: Load the data story and time series (medium weight)
        // Generate a data story based on the analyzed results
        setTimeout(async () => {
            try {
                await generateDataStory(query);
                
                // Update time series (important for overview)
                await updateTimeSeries(query).catch(error => {
                    console.error('Error updating time series:', error);
                    clearElement('timeseries-chart', '<p class="text-danger">Error loading time series data</p>');
                });
                
                // Description updates have been disabled
                
                // Phase 3: Now load the remaining heavier visualizations in the background
                // This allows the user to start interacting with the dashboard while heavy visualizations load
                setTimeout(async () => {
                    // Process remaining heavy visualizations in sequence to reduce load
                    try {
                        await updateTopics(query);
                        // Description updates have been disabled
                    } catch (error) {
                        console.error('Error updating topics:', error);
                        clearElement('topics-container', '<p class="text-danger">Error loading topics data</p>');
                    }
                    
                    // Network Analysis section removed
                    
                    try {
                        await updateCoordinatedBehavior();
                        // Description updates have been disabled
                    } catch (error) {
                        console.error('Error updating coordinated behavior:', error);
                        // Remove reference to coordinated-graph which no longer exists
                        clearElement('coordinated-groups', '<p class="text-danger">Error loading coordinated behavior data</p>');
                    }
                    
                    // Update Community Distribution pie chart
                    try {
                        await updateCommunityDistributionPieChart(query);
                        // Description updates have been disabled
                    } catch (error) {
                        console.error('Error updating community distribution:', error);
                        clearElement('community-distribution', '<p class="text-danger">Error loading community distribution data</p>');
                    }
                    
                    console.log('All visualizations loaded');
                }, 100);
                
            } catch (error) {
                console.error('Error in phase 2 loading:', error);
            }
        }, 10);
        
    } catch (error) {
        console.error('Error during analysis:', error);
        alert('Error performing analysis. Please check the console for details.');
        showLoading(false);
    }
});

// Initialize sliders and their value displays
document.getElementById('topics-count').addEventListener('input', (e) => {
    document.getElementById('topics-count-value').textContent = e.target.value;
});

document.getElementById('time-window').addEventListener('input', (e) => {
    document.getElementById('time-window-value').textContent = e.target.value;
});

document.getElementById('similarity-threshold').addEventListener('input', (e) => {
    document.getElementById('similarity-threshold-value').textContent = e.target.value;
});

// Button handlers for updating specific visualizations - only perform updates if sliders change
document.getElementById('update-topics-btn').addEventListener('click', async () => {
    showLoading(true);
    await updateTopics(activeQuery);
    // Update topics description again after refresh
    updateSectionDescription('topics', '#topics-description', {
        topicCount: document.getElementById('topics-count').value
    });
    showLoading(false);
});

// Initialize tab change listeners to ensure visualizations render properly on tab change
document.addEventListener('DOMContentLoaded', function() {
    const tabLinks = document.querySelectorAll('.nav-link');
    
    // When the page loads, make sure the active tab is properly set up
    const activeTab = document.querySelector('.nav-link.active');
    if (activeTab) {
        // Manually trigger the resize event to ensure proper initial rendering
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 100);
    }
    
    tabLinks.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(event) {
            // Trigger resize event to ensure visualizations are properly sized
            window.dispatchEvent(new Event('resize'));
            
            // If no analysis has been performed yet, no need to do anything else
            if (!analysisPerformed || !activeQuery) {
                return;
            }
            
            // Get the newly activated tab target
            const targetTab = event.target.getAttribute('data-bs-target');
            
            // Based on which tab is now active, ensure visualizations are properly rendered
            switch (targetTab) {
                case '#network-panel':
                    // If community distribution is empty, re-render it
                    if (document.getElementById('community-distribution').innerHTML === '') {
                        console.log('Re-rendering community distribution on tab change');
                        showLoading(true);
                        updateCommunityDistributionPieChart(activeQuery)
                            .catch(error => {
                                console.error('Error updating community distribution:', error);
                                document.getElementById('community-distribution').innerHTML = '<p class="text-danger">Error loading community distribution data</p>';
                            })
                            .finally(() => {
                                showLoading(false);
                            });
                    }
                    break;
                case '#timeseries-panel':
                    // If timeseries chart is empty, re-render it
                    if (document.getElementById('timeseries-chart').innerHTML === '' || 
                       document.getElementById('timeseries-chart').getBoundingClientRect().height < 10) {
                        console.log('Re-rendering time series on tab change');
                        showLoading(true);
                        updateTimeSeries(activeQuery)
                            .catch(error => {
                                console.error('Error updating time series:', error);
                                document.getElementById('timeseries-chart').innerHTML = '<p class="text-danger">Error loading time series data</p>';
                            })
                            .finally(() => {
                                showLoading(false);
                            });
                    }
                    break;
                case '#topics-panel':
                    // If topics container is empty, re-render it
                    if (document.getElementById('topics-container').innerHTML === '') {
                        console.log('Re-rendering topics on tab change');
                        showLoading(true);
                        updateTopics(activeQuery)
                            .catch(error => {
                                console.error('Error updating topics:', error);
                                document.getElementById('topics-container').innerHTML = '<p class="text-danger">Error loading topics data</p>';
                            })
                            .finally(() => {
                                showLoading(false);
                            });
                    }
                    break;
                case '#coordinated-panel':
                    // If coordinated groups is empty, re-render it
                    if (document.getElementById('coordinated-groups').innerHTML === '') {
                        console.log('Re-rendering coordinated behavior on tab change');
                        showLoading(true);
                        updateCoordinatedBehavior()
                            .catch(error => {
                                console.error('Error updating coordinated behavior:', error);
                                document.getElementById('coordinated-groups').innerHTML = '<p class="text-danger">Error loading coordinated behavior data</p>';
                            })
                            .finally(() => {
                                showLoading(false);
                            });
                    }
                    break;
                // Overview tab doesn't need special handling as it's loaded by default
            }
        });
    });
});

// Overview Section - AI Summary and Metrics
async function updateOverview(query) {
    try {
        // Update AI summary with enhanced insights
        const summaryResponse = await fetch(`/api/ai_summary?query=${encodeURIComponent(query)}`);
        if (summaryResponse.ok) {
            const summaryData = await summaryResponse.json();
            
            // Display model used and enhanced summary - the summary already contains HTML markup
            document.getElementById('ai-summary').innerHTML = summaryData.summary;
            
            // Update metrics with enhanced data
            const metrics = summaryData.metrics;
            
            // Basic metrics
            document.getElementById('total-posts').textContent = metrics.total_posts || '-';
            document.getElementById('unique-authors').textContent = metrics.unique_authors || '-';
            document.getElementById('avg-comments').textContent = typeof metrics.avg_comments === 'number' ? 
                metrics.avg_comments.toFixed(1) : metrics.avg_comments;
            document.getElementById('time-span').textContent = metrics.days_span || '-';
            
            // Update metrics description with actual data from the query
            const metricsDescriptionEl = document.getElementById('metrics-description');
            if (metricsDescriptionEl) {
                try {
                    const keywordsList = metrics.top_keywords && metrics.top_keywords.length > 0 
                        ? metrics.top_keywords.slice(0, 3).join(', ') 
                        : '';
                    
                    // Create a specific description based on actual query results
                    let description = `Showing ${metrics.total_posts} posts from ${metrics.unique_authors} unique authors`;
                    if (keywordsList) {
                        description += ` with popular keywords: ${keywordsList}`;
                    }
                    if (metrics.days_span) {
                        description += `. Data spans ${metrics.days_span} days`;
                        
                        // Add post frequency insight
                        const postsPerDay = (metrics.total_posts / metrics.days_span).toFixed(1);
                        description += ` (approximately ${postsPerDay} posts per day)`;
                    }
                    if (typeof metrics.avg_comments === 'number') {
                        description += ` with an average of ${metrics.avg_comments.toFixed(1)} comments per post`;
                        
                        // Contextual assessment of engagement level
                        if (metrics.avg_comments > 30) {
                            description += ` - indicating exceptionally high engagement`;
                        } else if (metrics.avg_comments > 15) {
                            description += ` - showing strong community interest`;
                        } else if (metrics.avg_comments > 5) {
                            description += ` - reflecting moderate discussion activity`;
                        }
                    }
                    
                    // Add author concentration insights if data available
                    if (metrics.unique_authors && metrics.total_posts) {
                        const postsPerAuthor = (metrics.total_posts / metrics.unique_authors).toFixed(1);
                        description += `. On average, each author contributed ${postsPerAuthor} posts`;
                        
                        // Community type assessment
                        if (postsPerAuthor > 3) {
                            description += `, suggesting a core group of dedicated contributors`;
                        } else if (postsPerAuthor < 1.2) {
                            description += `, indicating a diverse community with broad participation`;
                        }
                    }
                    
                    // Add sentiment if available
                    if (metrics.sentiment_distribution) {
                        const sentiments = metrics.sentiment_distribution;
                        const topSentiment = Object.entries(sentiments).sort((a, b) => b[1] - a[1])[0];
                        if (topSentiment) {
                            description += `. Conversation tone appears predominantly ${topSentiment[0].toLowerCase()}`;
                        }
                    }
                    
                    description += '.';
                    
                    // Update the DOM with the HTML-formatted description
                    metricsDescriptionEl.innerHTML = `
                        <div class="description-content">
                            <h4 class="section-heading mb-3 text-primary">Key Metrics</h4>
                            <p class="mb-2">${description}</p>
                        </div>
                    `;
                } catch (err) {
                    console.error('Failed to update metrics description:', err);
                    // Keep the existing message if there's an error
                }
            }
            
            // Create extended metrics display
            let metricsContainer = document.getElementById('metrics-container');
            
            // Clear any existing extended metrics (beyond the 4 main metrics)
            // Preserve only the main metrics row with the basic stats
            const mainMetricsRow = metricsContainer.querySelector('.row.g-3');
            metricsContainer.innerHTML = '';
            
            // Re-add the main metrics row
            if (mainMetricsRow) {
                metricsContainer.appendChild(mainMetricsRow);
            } else {
                // Create a new main metrics row if it doesn't exist
                const newMainRow = document.createElement('div');
                newMainRow.className = 'row g-3';
                newMainRow.innerHTML = `
                    <div class="col-6">
                        <div class="stat-card">
                            <div class="stat-value" id="total-posts">${metrics.total_posts || '-'}</div>
                            <div class="stat-label">Total Posts</div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="stat-card">
                            <div class="stat-value" id="unique-authors">${metrics.unique_authors || '-'}</div>
                            <div class="stat-label">Unique Authors</div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="stat-card">
                            <div class="stat-value" id="avg-comments">${typeof metrics.avg_comments === 'number' ? 
                                metrics.avg_comments.toFixed(1) : metrics.avg_comments}</div>
                            <div class="stat-label">Avg. Comments</div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="stat-card">
                            <div class="stat-value" id="time-span">${metrics.days_span || '-'}</div>
                            <div class="stat-label">Days Span</div>
                        </div>
                    </div>
                `;
                metricsContainer.appendChild(newMainRow);
            }
            
            // Add additional metrics rows if needed
            if (metrics.top_keywords && metrics.top_keywords.length > 0) {
                const keywordsRow = document.createElement('div');
                keywordsRow.className = 'row mt-3';
                keywordsRow.innerHTML = `
                    <div class="col-12">
                        <div class="stat-card">
                            <div class="stat-label">Top Keywords</div>
                            <div class="keyword-container">
                                ${metrics.top_keywords.map(kw => 
                                    `<span class="badge bg-primary me-1">${kw}</span>`
                                ).join('')}
                            </div>
                        </div>
                    </div>
                `;
                metricsContainer.appendChild(keywordsRow);
            }
            
            // Add top authors
            if (metrics.top_authors) {
                const authorsRow = document.createElement('div');
                authorsRow.className = 'row mt-3';
                authorsRow.innerHTML = `
                    <div class="col-12">
                        <div class="stat-card">
                            <div class="stat-label">Top Authors</div>
                            <div class="author-container">
                                <ul class="list-group list-group-flush small">
                                    ${Object.entries(metrics.top_authors).slice(0, 3).map(([author, count]) => 
                                        `<li class="list-group-item d-flex justify-content-between align-items-center">
                                            ${author}
                                            <span class="badge bg-primary rounded-pill">${count} posts</span>
                                        </li>`
                                    ).join('')}
                                </ul>
                            </div>
                        </div>
                    </div>
                `;
                metricsContainer.appendChild(authorsRow);
            }
            
            // Add engagement trend if available
            if (metrics.engagement_trend && Object.keys(metrics.engagement_trend).length > 0) {
                const engagementRow = document.createElement('div');
                engagementRow.className = 'row mt-3';
                engagementRow.innerHTML = `
                    <div class="col-12">
                        <div class="stat-card">
                            <div class="stat-label">Engagement Trend</div>
                            <div id="engagement-trend-chart" style="height: 200px;"></div>
                        </div>
                    </div>
                `;
                metricsContainer.appendChild(engagementRow);
                
                // Create enhanced chart for engagement trend with proper labels
                setTimeout(() => {
                    const trendData = Object.entries(metrics.engagement_trend).map(([date, value]) => ({
                        date: new Date(date),
                        value: value
                    })).sort((a, b) => a.date - b.date);
                    
                    if (trendData.length > 1) {
                        const trendWidth = document.getElementById('engagement-trend-chart').clientWidth;
                        const trendHeight = 200;
                        
                        // Define margins to accommodate axis labels - increase margins for better spacing
                        const margin = {top: 15, right: 35, bottom: 60, left: 60};
                        const width = trendWidth - margin.left - margin.right;
                        const height = trendHeight - margin.top - margin.bottom;
                        
                        // Clear previous chart if any
                        d3.select('#engagement-trend-chart').html('');
                        
                        const svg = d3.select('#engagement-trend-chart')
                            .append('svg')
                            .attr('width', trendWidth)
                            .attr('height', trendHeight)
                            .append('g')
                            .attr('transform', `translate(${margin.left},${margin.top})`);
                        
                        // Create X and Y scales with proper domains
                        const x = d3.scaleTime()
                            .domain(d3.extent(trendData, d => d.date))
                            .range([0, width]);
                        
                        const y = d3.scaleLinear()
                            .domain([0, d3.max(trendData, d => d.value) * 1.1]) // Add 10% padding at the top
                            .range([height, 0]);
                        
                        // Determine how many ticks to show based on data span
                        const timeSpan = trendData[trendData.length - 1].date - trendData[0].date;
                        const daySpan = timeSpan / (1000 * 60 * 60 * 24);
                        
                        // Choose appropriate date format and tick count based on date range
                        let tickCount = 4; // Default to 4 ticks
                        let dateFormat = '%b %Y'; // Default to month-year format
                        
                        if (daySpan <= 14) {
                            // For spans under 2 weeks, show day-month
                            dateFormat = '%d %b';
                            tickCount = Math.min(5, daySpan);
                        } else if (daySpan <= 60) {
                            // For spans under 60 days, show abbreviated month
                            dateFormat = '%b %d';
                            tickCount = Math.min(5, Math.ceil(daySpan / 7)); // About one tick per week
                        } else if (daySpan <= 365) {
                            // For spans under a year, show month only
                            dateFormat = '%b';
                            tickCount = Math.min(6, Math.ceil(daySpan / 30)); // About one tick per month
                        } else {
                            // For spans over a year, show month-year
                            tickCount = Math.min(6, Math.ceil(daySpan / 90)); // About one tick per quarter
                        }
                        
                        // Add X axis with properly formatted dates - fewer ticks, larger rotation
                        svg.append('g')
                            .attr('transform', `translate(0,${height})`)
                            .call(d3.axisBottom(x)
                                .ticks(tickCount)
                                .tickFormat(d3.timeFormat(dateFormat)))
                            .selectAll('text')
                            .style('text-anchor', 'end')
                            .style('font-size', '10px')
                            .attr('dx', '-.8em')
                            .attr('dy', '.15em')
                            .attr('transform', 'rotate(-45)'); // Increase rotation for better spacing
                        
                        // Add X axis label - position it lower for better spacing
                        svg.append('text')
                            .attr('transform', `translate(${width/2}, ${height + margin.bottom - 10})`)
                            .style('text-anchor', 'middle')
                            .style('font-size', '12px')
                            .style('fill', 'var(--text-secondary)')
                            .text('Date');
                        
                        // Add Y axis with grid lines - reduce the number of ticks
                        svg.append('g')
                            .call(d3.axisLeft(y)
                                .ticks(4) // Fewer ticks to avoid overlap
                                .tickFormat(d => d.toFixed(1)))
                            .call(g => g.selectAll('.tick line')
                                .clone()
                                .attr('x2', width)
                                .attr('stroke-opacity', 0.1));
                        
                        // Add Y axis label
                        svg.append('text')
                            .attr('transform', 'rotate(-90)')
                            .attr('y', -margin.left +5)
                            .attr('x', -height / 2)
                            .attr('dy', '1em')
                            .style('text-anchor', 'middle')
                            .style('font-size', '12px')
                            .style('fill', 'var(--text-secondary)')
                            .text('Avg. Comments');
                        
                        // Create a gradient for the line
                        const gradient = svg.append('defs')
                            .append('linearGradient')
                            .attr('id', 'engagement-gradient')
                            .attr('gradientUnits', 'userSpaceOnUse')
                            .attr('x1', 0)
                            .attr('y1', 0)
                            .attr('x2', 0)
                            .attr('y2', height);
                        
                        gradient.append('stop')
                            .attr('offset', '0%')
                            .attr('stop-color', 'var(--primary-color)')
                            .attr('stop-opacity', 1);
                        
                        gradient.append('stop')
                            .attr('offset', '100%')
                            .attr('stop-color', 'var(--primary-light)')
                            .attr('stop-opacity', 0.7);
                        
                        // Add area under the line
                        svg.append('path')
                            .datum(trendData)
                            .attr('fill', 'url(#engagement-gradient)')
                            .attr('fill-opacity', 0.3)
                            .attr('d', d3.area()
                                .x(d => x(d.date))
                                .y0(height)
                                .y1(d => y(d.value))
                            );
                        
                        // Add the line with smooth curve
                        svg.append('path')
                            .datum(trendData)
                            .attr('fill', 'none')
                            .attr('stroke', 'var(--primary-color)')
                            .attr('stroke-width', 2.5)
                            .attr('d', d3.line()
                                .curve(d3.curveMonotoneX)
                                .x(d => x(d.date))
                                .y(d => y(d.value))
                            );
                        
                        // Determine data point visibility based on density
                        // For dense datasets, only show a subset of points
                        let visiblePoints = trendData;
                        if (trendData.length > 20) {
                            // Create a subset of points to display based on significance
                            const significantPoints = [];
                            
                            // Always include first and last points
                            significantPoints.push(trendData[0]);
                            significantPoints.push(trendData[trendData.length - 1]);
                            
                            // Find local maxima and minima
                            for (let i = 1; i < trendData.length - 1; i++) {
                                const prev = trendData[i-1].value;
                                const curr = trendData[i].value;
                                const next = trendData[i+1].value;
                                
                                // Include if it's a peak or valley
                                if ((curr > prev && curr > next) || (curr < prev && curr < next)) {
                                    significantPoints.push(trendData[i]);
                                }
                                
                                // Include some regular samples for regular periods
                                if (i % Math.ceil(trendData.length / 15) === 0) {
                                    significantPoints.push(trendData[i]);
                                }
                            }
                            
                            // Sort points by date again
                            visiblePoints = significantPoints.sort((a, b) => a.date - b.date);
                        }
                        
                        // Add dots for selected data points
                        svg.selectAll('.dot')
                            .data(visiblePoints)
                            .enter()
                            .append('circle')
                            .attr('class', 'dot')
                            .attr('cx', d => x(d.date))
                            .attr('cy', d => y(d.value))
                            .attr('r', 3)
                            .attr('fill', 'var(--primary-color)')
                            .attr('stroke', '#fff')
                            .attr('stroke-width', 1.5)
                            .append('title')
                            .text(d => `${d3.timeFormat('%b %d, %Y')(d.date)}: ${d.value.toFixed(1)} comments`);
                    }
                }, 100);
            }
        }
        
        // Update top contributors (simplified version for overview)
        await updateContributorsOverview(query);
        
    } catch (error) {
        console.error('Error updating overview:', error);
    }
}

// Top Contributors - Overview Section
async function updateContributorsOverview(query) {
    const response = await fetch(`/api/top_contributors?query=${encodeURIComponent(query)}&limit=10`);
    const data = await response.json();
    
    const containerWidth = document.getElementById('contributors-overview').clientWidth;
    const margin = {top: 30, right: 30, bottom: 80, left: 60};
    const width = containerWidth - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;
    
    // Clear previous chart
    d3.select('#contributors-overview').html('');
    
    const svg = d3.select('#contributors-overview')
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Sort data by count in descending order
    data.sort((a, b) => b.count - a.count);
    
    const x = d3.scaleBand()
        .range([0, width])
        .domain(data.map(d => d.author))
        .padding(0.2);
    
    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.count) * 1.1]) // Add 10% padding at the top
        .range([height, 0]);
    
    // Add X axis with improved formatting for author names
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x))
        .selectAll('text')
        .attr('transform', 'translate(-10,0)rotate(-45)')
        .style('text-anchor', 'end')
        .style('font-size', '10px') // Smaller font for better fit
        .each(function(d) {
            // Truncate long author names
            const text = d3.select(this);
            const authorName = d;
            if (authorName.length > 12) {
                text.text(authorName.substring(0, 10) + '...');
                // Add full name as tooltip
                text.append('title').text(authorName);
            }
        });
    
    // Add X axis label
    svg.append('text')
        .attr('transform', `translate(${width/2}, ${height + margin.bottom - 10})`)
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', 'var(--text-secondary)')
        .text('Contributors');
    
    // Add Y axis with grid lines
    svg.append('g')
        .call(d3.axisLeft(y)
            .ticks(5)
            .tickFormat(d => Math.round(d)))
        .call(g => g.selectAll('.tick line')
            .clone()
            .attr('x2', width)
            .attr('stroke-opacity', 0.1));
    
    // Add Y axis label
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -margin.left + 15)
        .attr('x', -height / 2)
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('fill', 'var(--text-secondary)')
        .text('Number of Posts');
    
    // Add bars with a gradient and animation
    const barGradient = svg.append('defs')
        .append('linearGradient')
        .attr('id', 'bar-gradient')
        .attr('gradientUnits', 'userSpaceOnUse')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', 0)
        .attr('y2', height);
    
    barGradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', 'var(--primary-color)')
        .attr('stop-opacity', 1);
    
    barGradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', 'var(--primary-color)')
        .attr('stop-opacity', 0.7);
    
    // Add bars with animation and hover effect
    svg.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', d => x(d.author))
        .attr('y', height) // Start from bottom for animation
        .attr('width', x.bandwidth())
        .attr('height', 0) // Start with height 0 for animation
        .attr('fill', 'url(#bar-gradient)')
        .attr('rx', 2) // Slightly rounded corners
        // Add transition for bars
        .transition()
        .duration(800)
        .delay((d, i) => i * 50)
        .attr('y', d => y(d.count))
        .attr('height', d => height - y(d.count));
    
    // Add data labels for significant values
    svg.selectAll('.bar-label')
        .data(data.filter(d => d.count > d3.max(data, d => d.count) * 0.1)) // Only label bars with significant values
        .enter()
        .append('text')
        .attr('class', 'bar-label')
        .attr('x', d => x(d.author) + x.bandwidth() / 2)
        .attr('y', d => y(d.count) - 5)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('font-weight', 'bold')
        .style('fill', 'var(--text-primary)')
        .text(d => d.count)
        .style('opacity', 0) // Start invisible for animation
        .transition()
        .duration(800)
        .delay((d, i) => i * 50 + 300)
        .style('opacity', 1); // Fade in
    
    // Add hover interaction for bars
    svg.selectAll('rect')
        .on('mouseover', function(event, d) {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('fill', 'var(--primary-dark)');
            
            // Add tooltip on hover
            svg.append('text')
                .attr('class', 'tooltip-text')
                .attr('x', x(d.author) + x.bandwidth() / 2)
                .attr('y', y(d.count) - 15)
                .attr('text-anchor', 'middle')
                .style('font-size', '11px')
                .style('font-weight', 'bold')
                .style('fill', 'var(--text-primary)')
                .text(`${d.author}: ${d.count} posts`);
        })
        .on('mouseout', function() {
            d3.select(this)
                .transition()
                .duration(200)
                .attr('fill', 'url(#bar-gradient)');
            
            // Remove tooltip
            svg.selectAll('.tooltip-text').remove();
        });
}

// Word Cloud Visualization
async function updateWordCloud(query) {
    try {
        const response = await fetch(`/api/common_words?query=${encodeURIComponent(query)}&limit=100`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const words = await response.json();
        
        if (!Array.isArray(words) || words.length === 0) {
            console.warn('No word data available');
            return;
        }

        // Get the word cloud container and check if it exists
        const wordCloudContainer = d3.select('#word-cloud');
        if (wordCloudContainer.empty()) {
            console.warn('Word cloud container not found');
            return;
        }
        
        // Clear previous word cloud
        wordCloudContainer.html('');
        
        const width = wordCloudContainer.node().clientWidth;
        const height = 300;
        
        // Define vibrant color palette
        const colorPalette = [
            '#FF595E', '#FF924C', '#FFCA3A', '#8AC926', '#1982C4', 
            '#6A4C93', '#F94144', '#F3722C', '#F8961E', '#F9C74F',
            '#90BE6D', '#43AA8B', '#4D908E', '#577590', '#277DA1',
            '#E63946', '#F1FAEE', '#A8DADC', '#457B9D', '#1D3557'
        ];
        
        // Scale for word size
        const fontSize = d3.scaleLinear()
            .domain([0, d3.max(words, d => d.count)])
            .range([12, 60]);
            
        // Create SVG element
        const svg = d3.select('#word-cloud')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .style('border-radius', '12px')
            .style('overflow', 'hidden')
            .style('box-shadow', '0 4px 12px rgba(0, 0, 0, 0.05)')
            .style('transition', 'all 0.3s ease')
            .on('mouseenter', function() {
                d3.select(this)
                    .style('box-shadow', '0 6px 16px rgba(0, 0, 0, 0.1)')
                    .style('transform', 'translateY(-2px)');
            })
            .on('mouseleave', function() {
                d3.select(this)
                    .style('box-shadow', '0 4px 12px rgba(0, 0, 0, 0.05)')
                    .style('transform', 'translateY(0)');
            });
        
        // Add linear gradient definition
        const defs = svg.append('defs');
        const gradient = defs.append('linearGradient')
            .attr('id', 'word-cloud-background')
            .attr('x1', '0%')
            .attr('y1', '0%')
            .attr('x2', '100%')
            .attr('y2', '100%');
            
        gradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#f8f9fa')
            .attr('stop-opacity', 1);
            
        gradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#e9ecef')
            .attr('stop-opacity', 1);
        
        // Add background rectangle with gradient
        svg.append('rect')
            .attr('width', width)
            .attr('height', height)
            .attr('fill', 'url(#word-cloud-background)');
        
        // Create word cloud layout
        const wordCloudLayout = d3.layout.cloud()
            .size([width, height])
            .words(words.map(d => ({
                text: d.word,
                size: fontSize(d.count),
                count: d.count
            })))
            .padding(5)
            .rotate(() => (~~(Math.random() * 2) * 90))
            .fontSize(d => d.size)
            .on('end', draw);

        // Draw the word cloud
        let cloudGroup;
        function draw(words) {
            // Initialize cloudGroup
            cloudGroup = svg.append('g')
                .attr('transform', `translate(${width/2},${height/2})`)
                .attr('class', 'word-cloud-group');

            cloudGroup.selectAll('text')
                .data(words)
                .enter()
                .append('text')
                .style('font-size', d => `${d.size}px`)
                .style('font-family', d => d.font || '"Helvetica Neue", Arial, sans-serif')
                .style('font-weight', d => d.size > 30 ? 'bold' : (d.size > 20 ? 'semibold' : 'normal'))
                .style('fill', (d, i) => colorPalette[i % colorPalette.length])
                .style('cursor', 'pointer')
                .style('opacity', 0)
                .style('text-shadow', d => d.size > 25 ? '1px 1px 2px rgba(0,0,0,0.1)' : 'none')
                .attr('text-anchor', 'middle')
                .attr('transform', d => `translate(${d.x},${d.y})rotate(${d.rotate})`)
                .text(d => d.text)
                .on('mouseover', function(event, d) {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .style('font-size', `${d.size * 1.2}px`)
                        .style('font-weight', 'bold')
                        .style('fill', '#0d6efd');
                })
                .on('mouseout', function(event, d) {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .style('font-size', `${d.size}px`)
                        .style('font-weight', d.size > 30 ? 'bold' : (d.size > 20 ? 'semibold' : 'normal'))
                        .style('fill', (d, i) => colorPalette[i % colorPalette.length]);
                })
                .append('title')
                .text(d => {
                    const value = typeof d.value === 'number' ? d.value : (d.count || 0);
                    return `${d.text}: ${value}`;
                });

            // Animate words appearing with staggered timing
            cloudGroup.selectAll('text')
                .transition()
                .duration(600)
                .delay((d, i) => i * 30)
                .style('opacity', 1);
        }

        wordCloudLayout.start();
            
        // Create layout for topic cloud
        const topicCloudLayout = d3.layout.cloud()
            .size([width, height])
            .words(words.map(d => ({
                text: d.word, 
                size: fontSize(d.count),
                value: d.count,
                // Set all words to horizontal orientation (0 degrees)
                rotate: 0,
                // Add random font family selection for variety
                font: Math.random() > 0.7 ? 'Arial' : (Math.random() > 0.5 ? 'Helvetica' : 'Roboto')
            })))
            .padding(5)
            .fontSize(d => d.size)
            .font(d => d.font)
            .rotate(d => d.rotate)
            .spiral('archimedean')
            .on('end', draw);
        
        topicCloudLayout.start();
        
        function draw(words) {
            // Add words with animations and enhanced styling
            cloudGroup.selectAll('text')
                .data(words)
                .enter()
                .append('text')
                .style('font-size', d => `${d.size}px`)
                .style('font-family', d => d.font)
                .style('font-weight', d => d.size > 30 ? 'bold' : (d.size > 20 ? 'semibold' : 'normal'))
                .style('fill', (d, i) => colorPalette[i % colorPalette.length])
                .style('cursor', 'pointer')
                .style('opacity', 0) // Start with opacity 0 for fade-in animation
                .style('text-shadow', d => d.size > 25 ? '1px 1px 2px rgba(0,0,0,0.1)' : 'none')
                .attr('text-anchor', 'middle')
                .attr('transform', d => `translate(${d.x},${d.y}) rotate(${d.rotate})`)
                .text(d => d.text)
                .on('mouseover', function() {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .style('fill', '#0d6efd')
                        .style('font-size', function(d) { return `${d.size * 1.1}px`; });
                })
                .on('mouseout', function() {
                    d3.select(this)
                        .transition()
                        .duration(200)
                        .style('fill', function(d, i) { return colorPalette[i % colorPalette.length]; })
                        .style('font-size', function(d) { return `${d.size}px`; });
                })
                .append('title')
                .text(d => {
                    // Safe handling of value property to avoid the toFixed error
                    const value = typeof d.value === 'number' ? d.value : (d.value || 0);
                    return `${d.text}: ${value}`;
                });
            
            // Animate words appearing with staggered timing
            cloudGroup.selectAll('text')
                .transition()
                .duration(600)
                .delay((d, i) => i * 30)
                .style('opacity', 1);
        }
    } catch (error) {
        console.error('Error updating word cloud:', error);
        const wordCloudContainer = document.getElementById('word-cloud');
        if (wordCloudContainer) {
            wordCloudContainer.innerHTML = '<div class="alert alert-danger">Error loading word cloud data</div>';
        }
    }
}

// Time Series Visualization
async function updateTimeSeries(query) {
    const params = new URLSearchParams({
        query: query
    });
    
    const response = await fetch(`/api/timeseries?${params.toString()}`);
    const data = await response.json();
    
    // If there's no data, show a message
    if (!data || data.length === 0) {
        document.getElementById('timeseries-chart').innerHTML = '<div class="alert alert-info">No time series data available for this query.</div>';
        return;
    }
    
    const timeseriesElement = document.getElementById('timeseries-chart');
    const containerWidth = timeseriesElement.clientWidth || timeseriesElement.offsetWidth || 800; // Fallback width
    const margin = {top: 60, right: 80, bottom: 90, left: 70}; // Reduced right margin to maximize chart width
    const width = containerWidth - margin.left - margin.right;
    const height = 550 - margin.top - margin.bottom; // Further increased height for better spacing
    
    // Clear previous chart
    d3.select('#timeseries-chart').html('');
    
    // Create SVG with responsive width
    const svg = d3.select('#timeseries-chart')
        .append('svg')
        .attr('width', '100%') // Use 100% width to fill container
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Parse dates
    data.forEach(d => {
        d.date = new Date(d.date);
    });
    
    // Sort data by date
    data.sort((a, b) => a.date - b.date);
    
    // Calculate moving average (7-day window)
    const movingAvgWindow = Math.min(7, data.length);
    const movingAvgData = [];
    
    for (let i = 0; i < data.length; i++) {
        let sum = 0;
        let count = 0;
        
        for (let j = Math.max(0, i - Math.floor(movingAvgWindow/2)); 
             j <= Math.min(data.length - 1, i + Math.floor(movingAvgWindow/2)); j++) {
            sum += data[j].count;
            count++;
        }
        
        movingAvgData.push({
            date: data[i].date,
            count: sum / count
        });
    }
    
    // Find peak points (local maxima) - points with higher values than neighbors
    const peaks = [];
    for (let i = 1; i < data.length - 1; i++) {
        if (data[i].count > data[i-1].count && 
            data[i].count > data[i+1].count && 
            data[i].count > d3.mean(data, d => d.count) * 1.5) { // At least 50% above average
            peaks.push(data[i]);
        }
    }
    // Limit to top 3 peaks
    peaks.sort((a, b) => b.count - a.count);
    const topPeaks = peaks.slice(0, 3);
    
    // Identify overall trend
    const firstHalf = data.slice(0, Math.floor(data.length/2));
    const secondHalf = data.slice(Math.floor(data.length/2));
    const firstHalfAvg = d3.mean(firstHalf, d => d.count);
    const secondHalfAvg = d3.mean(secondHalf, d => d.count);
    const trendPercent = ((secondHalfAvg - firstHalfAvg) / firstHalfAvg * 100).toFixed(1);
    const trendDirection = secondHalfAvg > firstHalfAvg ? 'increasing' : 'decreasing';
    
    // Create scales
    const x = d3.scaleTime()
        .domain(d3.extent(data, d => d.date))
        .range([0, width]);
    
    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.count) * 1.1]) // Add space for annotations
        .range([height, 0]);
    
    // Create area gradient
    svg.append('linearGradient')
        .attr('id', 'area-gradient')
        .attr('gradientUnits', 'userSpaceOnUse')
        .attr('x1', 0).attr('y1', y(0))
        .attr('x2', 0).attr('y2', y(d3.max(data, d => d.count)))
        .selectAll('stop')
        .data([
            {offset: '0%', color: 'rgba(13, 110, 253, 0.1)'},
            {offset: '100%', color: 'rgba(13, 110, 253, 0.6)'}
        ])
        .enter().append('stop')
        .attr('offset', d => d.offset)
        .attr('stop-color', d => d.color);
    
    // Add the area
    svg.append('path')
        .datum(data)
        .attr('fill', 'url(#area-gradient)')
        .attr('stroke', 'none')
        .attr('d', d3.area()
            .x(d => x(d.date))
            .y0(height)
            .y1(d => y(d.count))
        );
    
    // Add the line
    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', '#0d6efd')
        .attr('stroke-width', 2)
        .attr('d', d3.line()
            .x(d => x(d.date))
            .y(d => y(d.count))
        );
    
    // Add the moving average line
    svg.append('path')
        .datum(movingAvgData)
        .attr('fill', 'none')
        .attr('stroke', '#dc3545')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .attr('d', d3.line()
            .x(d => x(d.date))
            .y(d => y(d.count))
        );
    
    // Add trend line
    if (data.length > 3) {
        // Simple linear regression for trend line
        const xValues = data.map((d, i) => i);
        const yValues = data.map(d => d.count);
        
        // Calculate linear regression (y = mx + b)
        const n = xValues.length;
        const sumX = xValues.reduce((a, b) => a + b, 0);
        const sumY = yValues.reduce((a, b) => a + b, 0);
        const sumXY = xValues.reduce((a, b, i) => a + b * yValues[i], 0);
        const sumXX = xValues.reduce((a, b) => a + b * b, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // Create trend line
        const trendData = [
            { x: xValues[0], y: slope * xValues[0] + intercept },
            { x: xValues[xValues.length - 1], y: slope * xValues[xValues.length - 1] + intercept }
        ];
        
        // Draw trend line
        svg.append('line')
            .attr('x1', x(data[0].date))
            .attr('y1', y(trendData[0].y))
            .attr('x2', x(data[data.length - 1].date))
            .attr('y2', y(trendData[1].y))
            .attr('stroke', '#20c997')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '10,5')
            .attr('opacity', 0.7);
    }
    
    // Add data points with enhanced interaction
    svg.selectAll('.dot')
        .data(data)
        .enter().append('circle')
        .attr('class', 'dot')
        .attr('cx', d => x(d.date))
        .attr('cy', d => y(d.count))
        .attr('r', 4)
        .attr('fill', '#0d6efd')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            d3.select(this)
              .transition()
              .duration(100)
              .attr('r', 6);
              
            // Add tooltip
            const tooltipX = event.pageX;
            const tooltipY = event.pageY - 40;
            
            d3.select('body')
              .append('div')
              .attr('class', 'timeseries-tooltip')
              .style('position', 'absolute')
              .style('left', `${tooltipX}px`)
              .style('top', `${tooltipY}px`)
              .style('background', 'rgba(0,0,0,0.8)')
              .style('color', 'white')
              .style('padding', '5px 10px')
              .style('border-radius', '5px')
              .style('font-size', '12px')
              .style('pointer-events', 'none')
              .style('z-index', 1000)
              .html(`Date: ${d.date.toLocaleDateString()}<br>Posts: ${d.count}`);
        })
        .on('mouseout', function() {
            d3.select(this)
              .transition()
              .duration(100)
              .attr('r', 4);
              
            d3.selectAll('.timeseries-tooltip').remove();
        })
        .append('title')
        .text(d => `Date: ${d.date.toLocaleDateString()}\nPosts: ${d.count}`);
    
    // Highlight peak points with enhanced styling
    svg.selectAll('.peak')
        .data(topPeaks)
        .enter().append('circle')
        .attr('class', 'peak')
        .attr('cx', d => x(d.date))
        .attr('cy', d => y(d.count))
        .attr('r', 8)
        .attr('fill', '#dc3545')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            d3.select(this)
              .transition()
              .duration(100)
              .attr('r', 10);
              
            // Add tooltip
            const tooltipX = event.pageX;
            const tooltipY = event.pageY - 40;
            
            d3.select('body')
              .append('div')
              .attr('class', 'timeseries-tooltip peak-tooltip')
              .style('position', 'absolute')
              .style('left', `${tooltipX}px`)
              .style('top', `${tooltipY}px`)
              .style('background', 'rgba(220,53,69,0.9)')
              .style('color', 'white')
              .style('padding', '5px 10px')
              .style('border-radius', '5px')
              .style('font-size', '12px')
              .style('pointer-events', 'none')
              .style('z-index', 1000)
              .html(`<strong>Peak!</strong><br>Date: ${d.date.toLocaleDateString()}<br>Posts: ${d.count}`);
        })
        .on('mouseout', function() {
            d3.select(this)
              .transition()
              .duration(100)
              .attr('r', 8);
              
            d3.selectAll('.timeseries-tooltip').remove();
        })
        .append('title')
        .text(d => `Peak: ${d.date.toLocaleDateString()}\nPosts: ${d.count}`);
    
    // Add peak annotations with improved visibility
    topPeaks.forEach((peak, i) => {
        // Only add text annotations for the top peak to avoid clutter
        if (i === 0) {
            // Create text with white outline for better readability
            const peakLabel = svg.append('text')
                .attr('x', x(peak.date))
                .attr('y', y(peak.count) - 20)
                .attr('text-anchor', 'middle')
                .style('font-size', '12px')
                .style('font-weight', 'bold')
                .style('paint-order', 'stroke')
                .style('stroke', 'white')
                .style('stroke-width', '3px')
                .style('fill', '#dc3545')
                .text(`Peak: ${peak.count} posts`);
        }
    });
    
    // Add X axis with improved formatting
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x)
            .ticks(Math.min(data.length, width > 600 ? 10 : 5))
            .tickFormat(d3.timeFormat('%b %d')))
        .selectAll('text')
        .style('text-anchor', 'end')
        .attr('dx', '-.8em')
        .attr('dy', '.15em')
        .attr('transform', 'rotate(-45)')
        .style('font-size', '11px');
    
    // Add Y axis with improved tick formatting
    svg.append('g')
        .call(d3.axisLeft(y)
            .ticks(8)
            .tickFormat(d => {
                if (d >= 1000) return d3.format(',.1k')(d);
                return d;
            }))
        .selectAll('text')
        .style('font-size', '11px');
    
    // Add title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -margin.top / 2 + 5)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .style('font-weight', 'bold')
        .text(`Post Frequency for "${query}"`);
    
    // Add trend indicator below title with sufficient spacing
    const trendColor = trendDirection === 'increasing' ? '#20c997' : '#dc3545';
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -margin.top / 2 + 30) // Positioned well below title
        .attr('text-anchor', 'middle')
        .style('font-size', '13px')
        .style('font-weight', 'bold')
        .style('fill', trendColor)
        .text(`Trend: ${trendDirection} (${trendPercent}%)`);
    
    // Add legend with positioning for full-width chart
    const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${width - 10}, 40)`);
    
    // Data points
    legend.append('circle')
        .attr('cx', 0)
        .attr('cy', 0)
        .attr('r', 4)
        .attr('fill', '#0d6efd')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1);
    
    legend.append('text')
        .attr('x', 10)
        .attr('y', 4)
        .style('font-size', '12px')
        .text('Daily posts')
        .attr('alignment-baseline', 'middle');
    
    // Moving average
    legend.append('line')
        .attr('x1', -10)
        .attr('y1', 25)
        .attr('x2', 5)
        .attr('y2', 25)
        .attr('stroke', '#dc3545')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');
    
    legend.append('text')
        .attr('x', 10)
        .attr('y', 29)
        .style('font-size', '12px')
        .text('7-day average')
        .attr('alignment-baseline', 'middle');
    
    // Trend line
    legend.append('line')
        .attr('x1', -10)
        .attr('y1', 50)
        .attr('x2', 5)
        .attr('y2', 50)
        .attr('stroke', '#20c997')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '10,5');
    
    legend.append('text')
        .attr('x', 10)
        .attr('y', 54)
        .style('font-size', '12px')
        .text('Trend line')
        .attr('alignment-baseline', 'middle');
    
    // Add Y axis label
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', 0 - margin.left + 15)
        .attr('x', 0 - (height / 2))
        .attr('dy', '1em')
        .style('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '500')
        .attr('fill', 'var(--text-primary)')
        .text('Number of Posts');
    
    // Add X axis label
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', height + margin.bottom - 15)
        .style('text-anchor', 'middle')
        .style('font-size', '14px')
        .style('font-weight', '500')
        .attr('fill', 'var(--text-primary)')
        .text('Date');
    
    // Force a redraw if needed (helps with rendering issues)
    setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
    
    // Store time series data and peaks in global variable for later use
    window.timeSeriesData = {
        raw: data,
        peaks: topPeaks,
        trend: {
            direction: trendDirection,
            percent: trendPercent
        }
    };
}

// Top Contributors Visualization - Full Page
async function updateContributors(query) {
    const response = await fetch(`/api/top_contributors?query=${encodeURIComponent(query)}&limit=20`);
    const data = await response.json();
    
    // Create a pie chart based on the top contributors
    const width = 600;
    const height = 400;
    const radius = Math.min(width, height) / 2;
    
    // Clear previous chart
    d3.select('#contributors').html('');
    
    const svg = d3.select('#contributors')
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`);
    
    // Color scale
    const color = d3.scaleOrdinal(d3.schemeCategory10);
    
    // Compute the position of each group on the pie:
    const pie = d3.pie()
        .value(d => d.count)
        .sort(null);
    
    const data_ready = pie(data);
    
    // Shape helper to build arcs:
    const arcGenerator = d3.arc()
        .innerRadius(0)
        .outerRadius(radius);
    
    // Build the pie chart
    svg.selectAll('slices')
        .data(data_ready)
        .enter()
        .append('path')
        .attr('d', arcGenerator)
        .attr('fill', (d, i) => color(i))
        .attr('stroke', 'white')
        .style('stroke-width', '2px')
        .style('opacity', 0.7);
    
    // Add labels
    svg.selectAll('slices')
        .data(data_ready)
        .enter()
        .append('text')
        .text(d => d.data.author)
        .attr('transform', d => {
            const pos = arcGenerator.centroid(d);
            const midAngle = Math.atan2(pos[1], pos[0]);
            const x = Math.cos(midAngle) * (radius + 20);
            const y = Math.sin(midAngle) * (radius + 20);
            return `translate(${x},${y})`;
        })
        .style('text-anchor', d => {
            const pos = arcGenerator.centroid(d);
            return (pos[0] > 0) ? 'start' : 'end';
        })
        .style('font-size', '12px')
        .style('font-weight', 'bold');
    
    // Add polylines between pie and labels
    svg.selectAll('allPolylines')
        .data(data_ready)
        .enter()
        .append('polyline')
        .attr('stroke', 'black')
        .style('fill', 'none')
        .attr('stroke-width', 1)
        .attr('points', d => {
            const posA = arcGenerator.centroid(d);  // line insertion in the slice
            const posB = arcGenerator.centroid(d);  // line break: we use the same y as posA
            const posC = arcGenerator.centroid(d);  // Label position = almost the same as posB
            const midAngle = Math.atan2(posA[1], posA[0]);
            const posB_x = Math.cos(midAngle) * (radius + 10);
            const posB_y = Math.sin(midAngle) * (radius + 10);
            const posC_x = Math.cos(midAngle) * (radius + 20);
            const posC_y = Math.sin(midAngle) * (radius + 20);
            return [posA, [posB_x, posB_y], [posC_x, posC_y]];
        });
    
    // Add title
    svg.append('text')
        .attr('x', 0)
        .attr('y', -height/2 + 20)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .text(`Top Contributors for "${query}"`);
}

// Network Visualization - Simplified version after removing Network Analysis section
async function updateNetwork(query) {
    console.log('Network analysis skipped - section has been removed');
    // No longer tries to access network-graph element
    return Promise.resolve();
}

// Topic Analysis Visualization - Enhanced version
async function updateTopics(query) {
    const topicsCount = document.getElementById('topics-count').value;
    const response = await fetch(`/api/topics?n_topics=${topicsCount}&query=${encodeURIComponent(query)}`);
    if (!response.ok) {
        throw new Error(`Topics request failed with status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Check if we have valid data
    if (!data || !data.topics || data.topics.length === 0) {
        document.getElementById('topics-container').innerHTML = '<div class="alert alert-info">No topic data available for this query.</div>';
        return;
    }
    
    // Clear previous visualization
    document.getElementById('topics-container').innerHTML = '';
    
    // Create the enhanced topic visualization
    const topicsContainer = document.getElementById('topics-container');
    
    // Add summary metrics
    const metricsHtml = `
        <div class="card mb-4">
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4 text-center">
                        <div class="h2">${data.n_docs_analyzed}</div>
                        <div class="text-muted">Documents Analyzed</div>
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="h2">${data.topics.length}</div>
                        <div class="text-muted">Topics Identified</div>
                    </div>
                    <div class="col-md-4 text-center">
                        <div class="h2">${data.coherence_score ? (data.coherence_score * 100).toFixed(1) : '0'}%</div>
                        <div class="text-muted">Coherence Score</div>
                    </div>
                </div>
            </div>
        </div>
    `;
    topicsContainer.innerHTML = metricsHtml;
    
    // PERFORMANCE OPTIMIZATION: Simplified topic evolution chart
    // Create topic evolution chart only if there's a reasonable amount of data
    if (data.topic_evolution && Object.keys(data.topic_evolution).length > 0) {
        // Count total data points to decide on optimization level
        let totalDataPoints = 0;
        let maxTopics = 0;
        
        for (const [topicId, dateCounts] of Object.entries(data.topic_evolution)) {
            maxTopics++;
            totalDataPoints += Object.keys(dateCounts).length;
        }
        
        // Only show evolution chart if there's not too much data
        if (totalDataPoints < 200 && maxTopics <= 5) {
            const evolutionContainer = document.createElement('div');
            evolutionContainer.className = 'card mb-4';
            evolutionContainer.innerHTML = `
                <div class="card-header">
                    <h5 class="mb-0">Topic Evolution Over Time</h5>
                </div>
                <div class="card-body">
                    <div id="topic-evolution-chart-mini" style="height: 350px;"></div>
                </div>
            `;
            topicsContainer.appendChild(evolutionContainer);
            
            // Process data for the chart
            setTimeout(() => {
                const evolutionChart = document.getElementById('topic-evolution-chart-mini');
                const chartWidth = evolutionChart.clientWidth;
                const chartHeight = 350;
                
                // Process evolution data for visualization
                const timeData = {};
                const allDates = new Set();
                const topicIds = [];
                
                // Collect all dates and topic ids
                for (const [topicId, dateCounts] of Object.entries(data.topic_evolution)) {
                    topicIds.push(topicId);
                    for (const date of Object.keys(dateCounts)) {
                        allDates.add(date);
                    }
                }
                
                // Sort dates chronologically
                const sortedDates = Array.from(allDates).sort();
                
                // Create data points for each topic over time
                const chartData = [];
                topicIds.forEach((topicId, i) => {
                    const topicData = {
                        name: `Topic ${topicId.split('_')[1]}`,
                        values: []
                    };
                    
                    sortedDates.forEach(date => {
                        topicData.values.push({
                            date: new Date(date),
                            value: data.topic_evolution[topicId][date] || 0
                        });
                    });
                    
                    chartData.push(topicData);
                });
                
                // Create the chart
                const svg = d3.select('#topic-evolution-chart-mini')
                    .append('svg')
                    .attr('width', chartWidth)
                    .attr('height', chartHeight);
                
                // Setup scales
                const x = d3.scaleTime()
                    .domain(d3.extent(sortedDates, d => new Date(d)))
                    .range([50, chartWidth - 20]);
                
                // Find max value across all topics and dates
                const maxValue = d3.max(chartData, d => d3.max(d.values, v => v.value));
                
                const y = d3.scaleLinear()
                    .domain([0, maxValue])
                    .range([chartHeight - 30, 20]);
                
                // Add axes
                svg.append('g')
                    .attr('transform', `translate(0,${chartHeight - 30})`)
                    .call(d3.axisBottom(x).ticks(5).tickFormat(d3.timeFormat('%b %d')));
                
                svg.append('g')
                    .attr('transform', 'translate(50,0)')
                    .call(d3.axisLeft(y));
                
                // Add lines for each topic
                const color = d3.scaleOrdinal(d3.schemeCategory10);
                const line = d3.line()
                    .x(d => x(d.date))
                    .y(d => y(d.value))
                    // PERFORMANCE OPTIMIZATION: Use linear curves instead of basis for better performance
                    .curve(d3.curveLinear);
                
                chartData.forEach((topic, i) => {
                    svg.append('path')
                        .datum(topic.values)
                        .attr('fill', 'none')
                        .attr('stroke', color(i))
                        .attr('stroke-width', 2)
                        .attr('d', line);
                    
                    // PERFORMANCE OPTIMIZATION: Only add text for significant values
                    // Add topic name at the end of the line
                    const lastDataPoint = topic.values[topic.values.length - 1];
                    if (lastDataPoint && lastDataPoint.value > maxValue * 0.1) {
                        svg.append('text')
                            .attr('x', x(lastDataPoint.date) + 5)
                            .attr('y', y(lastDataPoint.value))
                            .attr('fill', color(i))
                            .attr('font-size', '12px')
                            .text(topic.name);
                    }
                });
            }, 100);
        } else {
            // Display a simplified message for large datasets
            const evolutionContainer = document.createElement('div');
            evolutionContainer.className = 'card mb-4';
            evolutionContainer.innerHTML = `
                <div class="card-header">
                    <h5 class="mb-0">Topic Evolution</h5>
                </div>
                <div class="card-body">
                    <p class="text-muted">Topic evolution details are available but simplified for performance reasons.</p>
                    <p>Topics identified: ${maxTopics}, Total data points: ${totalDataPoints}</p>
                </div>
            `;
            topicsContainer.appendChild(evolutionContainer);
        }
    }
    
    // PERFORMANCE OPTIMIZATION: Create topic cards with lazy-loaded visualizations
    // Limit to 5 topic cards for performance
    const visibleTopics = Math.min(data.topics.length, 5);
    
    // Create cards for the first 5 topics
    for (let i = 0; i < visibleTopics; i++) {
        const topic = data.topics[i];
        const topicCard = document.createElement('div');
        topicCard.className = 'card mb-3';
        
        // Create simplified word cloud data for this topic
        const wordCloudData = topic.word_weight_pairs.slice(0, 20).map(pair => ({
            text: pair.word,
            size: 14 + (pair.weight * 1.5), // Scale weight to font size
            weight: pair.weight
        }));
        
        // Topic card with word cloud and representative docs
        topicCard.innerHTML = `
            <div class="card-header bg-light">
                <h5 class="mb-0">Topic ${topic.topic_id + 1}</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-7">
                        <div id="topic-cloud-${topic.topic_id}" class="word-cloud" style="height: 200px;">
                            <div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading word cloud...</div>
                        </div>
                    </div>
                    <div class="col-md-5">
                        <h6>Key Words</h6>
                        <div class="keyword-list">
                            ${topic.word_weight_pairs.slice(0, 8).map(pair => 
                                `<div class="d-flex justify-content-between mb-1">
                                    <span>${pair.word}</span>
                                    <div class="progress flex-grow-1 mx-2" style="height: 6px; margin-top: 8px;">
                                        <div class="progress-bar" style="width: ${pair.weight}%"></div>
                                    </div>
                                    <span class="badge bg-secondary">${pair.weight.toFixed(1)}%</span>
                                </div>`
                            ).join('')}
                        </div>
                    </div>
                </div>
                ${topic.representative_docs && topic.representative_docs.length > 0 ? `
                    <div class="mt-3">
                        <h6>Representative Posts</h6>
                        <ul class="list-group">
                            ${topic.representative_docs.slice(0, 2).map(doc => `
                                <li class="list-group-item">
                                    <div class="small text-muted">r/${doc.subreddit} • ${new Date(doc.created_utc).toLocaleDateString()} • u/${doc.author}</div>
                                    <div>${doc.title}</div>
                                    <div class="small mt-1">Topic relevance: ${(doc.topic_probability * 100).toFixed(1)}%</div>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;
        
        topicsContainer.appendChild(topicCard);
    }
    
    // If there are more topics, add a message
    if (data.topics.length > visibleTopics) {
        const moreTopicsMessage = document.createElement('div');
        moreTopicsMessage.className = 'alert alert-info';
        moreTopicsMessage.textContent = `Showing ${visibleTopics} of ${data.topics.length} topics for performance reasons. Adjust the number of topics and click "Update Topics" to see different topics.`;
        topicsContainer.appendChild(moreTopicsMessage);
    }
    
    // PERFORMANCE OPTIMIZATION: Lazy-load word clouds with IntersectionObserver
    // Only render the word clouds when they become visible
    if ('IntersectionObserver' in window) {
        const cloudObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const targetId = entry.target.id;
                    const topicId = targetId.split('-').pop();
                    const topic = data.topics.find(t => t.topic_id == topicId);
                    
                    if (topic) {
                        renderWordCloud(topicId, topic.word_weight_pairs.slice(0, 20));
                        observer.unobserve(entry.target);
                    }
                }
            });
        }, {
            threshold: 0.1
        });
        
        // Observe all word clouds
        for (let i = 0; i < visibleTopics; i++) {
            const cloudContainer = document.getElementById(`topic-cloud-${data.topics[i].topic_id}`);
            if (cloudContainer) {
                cloudObserver.observe(cloudContainer);
            }
        }
    } else {
        // Fallback for browsers without IntersectionObserver
        // Render word clouds with a delay between each
        for (let i = 0; i < visibleTopics; i++) {
            const topicId = data.topics[i].topic_id;
            setTimeout(() => {
                renderWordCloud(topicId, data.topics[i].word_weight_pairs.slice(0, 20));
            }, i * 300); // Stagger loading
        }
    }
    
    // Helper function to render a word cloud
    function renderWordCloud(topicId, wordWeightPairs) {
        const cloudContainer = document.getElementById(`topic-cloud-${topicId}`);
        if (!cloudContainer) return;
        
        // Clear loading indicator
        cloudContainer.innerHTML = '';
        
        const width = cloudContainer.clientWidth;
        const height = cloudContainer.clientHeight || 200;
        
        // Process words data
        const wordCloudData = wordWeightPairs.map(pair => ({
            text: pair.word,
            size: 14 + (pair.weight * 1.5),
            weight: pair.weight
        }));
        
        // PERFORMANCE OPTIMIZATION: Simplified word cloud with fewer words
        // Create layout with fewer words and simplified options
        const summaryCloudLayout = d3.layout.cloud()
            .size([width, height])
            .words(wordCloudData)
            .padding(3)
            .rotate(() => 0) // No rotation for better performance
            .font('Arial')
            .fontSize(d => d.size)
            .on('end', words => {
                // Color scale
                const color = d3.scaleLinear()
                    .domain([0, 50, 100])
                    .range(['#6c757d', '#0d6efd', '#dc3545']);
                
                d3.select(`#topic-cloud-${topicId}`)
                    .append('svg')
                    .attr('width', width)
                    .attr('height', height)
                    .append('g')
                    .attr('transform', `translate(${width/2},${height/2})`)
                    .selectAll('text')
                    .data(words)
                    .enter()
                    .append('text')
                    .style('font-size', d => `${d.size}px`)
                    .style('font-family', 'Arial')
                    .style('font-weight', d => d.weight > 20 ? 'bold' : 'normal')
                    .style('fill', d => color(d.weight))
                    .attr('text-anchor', 'middle')
                    .attr('transform', d => `translate(${d.x},${d.y}) rotate(${d.rotate})`)
                    .text(d => d.text)
                    .on('mouseover', function() {
                        d3.select(this)
                            .transition()
                            .duration(200)
                            .style('fill', '#0d6efd')
                            .style('font-size', d => `${d.size * 1.1}px`);
                    })
                    .on('mouseout', function(event, d) {
                        d3.select(this)
                            .transition()
                            .duration(200)
                            .style('fill', (d, i) => colorPalette[i % colorPalette.length])
                            .style('font-size', d => `${d.size}px`);
                    })
                    .append('title')
                    .text(d => {
                        // Safe handling of weight property to avoid the toFixed error
                        const weight = typeof d.weight === 'number' ? d.weight.toFixed(1) : (d.weight || 0);
                        return `${d.text}: ${weight}%`;
                    });
                
                // Animate words appearing with staggered timing
                cloudGroup.selectAll('text')
                    .transition()
                    .duration(600)
                    .delay((d, i) => i * 30)
                    .style('opacity', 1);
            });
        
        summaryCloudLayout.start();
    }
}

// Coordinated Behavior Analysis - Enhanced version
async function updateCoordinatedBehavior() {
    try {
        const timeWindow = document.getElementById('time-window').value;
        const similarityThreshold = document.getElementById('similarity-threshold').value;
        const query = activeQuery; // Use the active query for filtering
        
        const response = await fetch(`/api/coordinated?time_window=${timeWindow}&similarity_threshold=${similarityThreshold}&query=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        // Update stats with enhanced metrics
        const metrics = data.metrics || { 
            total_groups: 0, 
            total_authors: 0 
        };
        
        // Safely update DOM elements after checking they exist
        const totalGroupsEl = document.getElementById('total-groups');
        if (totalGroupsEl) totalGroupsEl.textContent = metrics.total_groups;
        
        const totalAuthorsEl = document.getElementById('total-authors');
        if (totalAuthorsEl) totalAuthorsEl.textContent = metrics.total_authors;
        
        // Create enhanced metrics display
        const statsContainer = document.getElementById('coordinated-stats');
        if (!statsContainer) {
            console.warn('Coordinated stats container not found');
        } else {
            statsContainer.innerHTML = `
                <div class="row mb-3">
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="stat-value">${metrics.total_groups}</div>
                            <div class="stat-label">Coordinated Groups</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="stat-value">${metrics.total_authors}</div>
                            <div class="stat-label">Authors Involved</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="stat-value">${metrics.avg_group_size ? metrics.avg_group_size.toFixed(1) : '-'}</div>
                            <div class="stat-label">Avg Group Size</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="stat-value">${metrics.authors_involved_percentage ? metrics.authors_involved_percentage.toFixed(1) + '%' : '-'}</div>
                            <div class="stat-label">% of All Authors</div>
                        </div>
                    </div>
                </div>
                <div class="alert alert-info">
                    <i class="bi bi-info-circle-fill"></i> 
                    Analysis performed with ${metrics.time_window_seconds}s time window and ${metrics.similarity_threshold} similarity threshold.
                </div>
            `;
        }
        
        // Safely clear previous visualizations
        const coordGroupsEl = document.getElementById('coordinated-groups');
        if (coordGroupsEl) d3.select('#coordinated-groups').html('');
        
        // Validate data structure to avoid errors
        if (!data.network || !data.network.nodes || !data.network.nodes.length) {
            if (coordGroupsEl) coordGroupsEl.innerHTML = '<div class="alert alert-info">No coordinated behavior detected with current parameters. Try adjusting the time window or similarity threshold parameters.</div>';
            return;
        }
        
        // Update the avg-group-size and authors-percentage elements if they exist
        const avgGroupSizeEl = document.getElementById('avg-group-size');
        if (avgGroupSizeEl) avgGroupSizeEl.textContent = metrics.avg_group_size ? metrics.avg_group_size.toFixed(1) : '0';
        
        const authorsPercentageEl = document.getElementById('authors-percentage');
        if (authorsPercentageEl) authorsPercentageEl.textContent = metrics.authors_involved_percentage ? metrics.authors_involved_percentage.toFixed(1) + '%' : '0%';
        
        // Skip network graph rendering since we've removed that section
        
        // 2. Render the enhanced coordinated groups panel
        const groupsContainer = d3.select('#coordinated-groups');
        
        if (!groupsContainer.empty()) {
            // Add search functionality
            groupsContainer.html(`
                <div class="mb-3">
                    <input type="text" class="form-control" id="group-search" placeholder="Search in coordinated groups...">
                </div>
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Group ID</th>
                                <th>Size</th>
                                <th>Time Span</th>
                                <th>Authors</th>
                                <th>Details</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.groups.map((group, idx) => {
                                const timeSpanMinutes = Math.ceil(group.time_span / 60);
                                return `
                                    <tr data-group-id="${group.group_id}">
                                        <td><span class="badge bg-primary">#${group.group_id}</span></td>
                                        <td>${group.size} posts</td>
                                        <td>${timeSpanMinutes} min</td>
                                        <td>${group.unique_authors} authors</td>
                                        <td>
                                            <button class="btn btn-sm btn-outline-secondary toggle-details">
                                                <i class="bi bi-chevron-down"></i> View Details
                                            </button>
                                        </td>
                                    </tr>
                                    <tr class="group-details" style="display: none;">
                                        <td colspan="5">
                                            <div class="card">
                                                <div class="card-body">
                                                    <h6 class="card-subtitle mb-2 text-muted">Posts in this group:</h6>
                                                    <div class="coordinated-posts mt-3">
                                                        ${group.posts.map(post => {
                                                            const postTime = new Date(post.created_utc).toLocaleString();
                                                            return `
                                                                <div class="coordinated-group mb-3">
                                                                    <h6>${post.title}</h6>
                                                                    <div class="small text-muted mb-2">
                                                                        Posted by u/${post.author} at ${postTime}
                                                                    </div>
                                                                    <div class="coordinated-content">
                                                                        ${post.selftext ? `<p>${post.selftext}</p>` : '<p><em>No text content</em></p>'}
                                                                    </div>
                                                                    <div class="mt-2">
                                                                        <a href="${post.url}" target="_blank" class="btn btn-sm btn-link">
                                                                            <i class="bi bi-box-arrow-up-right"></i> View Original
                                                                        </a>
                                                                        ${post.similarity_score ? 
                                                                            `<span class="badge bg-info text-dark">
                                                                                Similarity: ${Math.round(post.similarity_score * 100)}%
                                                                            </span>` : ''
                                                                        }
                                                                    </div>
                                                                </div>
                                                            `;
                                                        }).join('')}
                                                    </div>
                                                </div>
                                            </div>
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            `);
            
            // Add toggle behavior for group details
            document.querySelectorAll('.toggle-details').forEach(button => {
                button.addEventListener('click', function() {
                    const row = this.closest('tr');
                    const detailsRow = row.nextElementSibling;
                    const isHidden = detailsRow.style.display === 'none';
                    
                    // Hide all other detail rows
                    document.querySelectorAll('.group-details').forEach(r => {
                        r.style.display = 'none';
                    });
                    
                    document.querySelectorAll('.toggle-details i').forEach(icon => {
                        icon.className = 'bi bi-chevron-down';
                    });
                    
                    // Toggle this detail row
                    if (isHidden) {
                        detailsRow.style.display = 'table-row';
                        this.querySelector('i').className = 'bi bi-chevron-up';
                        this.textContent = ' Hide Details';
                        this.prepend(this.querySelector('i'));
                    } else {
                        detailsRow.style.display = 'none';
                        this.querySelector('i').className = 'bi bi-chevron-down';
                        this.textContent = ' View Details';
                        this.prepend(this.querySelector('i'));
                    }
                });
            });
            
            // Add search functionality
            const searchInput = document.getElementById('group-search');
            if (searchInput) {
                searchInput.addEventListener('input', function() {
                    const searchTerm = this.value.toLowerCase();
                    
                    document.querySelectorAll('#coordinated-groups tbody tr').forEach(row => {
                        if (!row.classList.contains('group-details')) {
                            const rowText = row.textContent.toLowerCase();
                            const detailsRow = row.nextElementSibling;
                            
                            if (rowText.includes(searchTerm) || detailsRow.textContent.toLowerCase().includes(searchTerm)) {
                                row.style.display = '';
                                detailsRow.style.display = searchTerm && searchTerm.trim() !== '' ? 'table-row' : 'none';
                                
                                if (searchTerm && searchTerm.trim() !== '') {
                                    row.querySelector('.toggle-details i').className = 'bi bi-chevron-up';
                                    row.querySelector('.toggle-details').textContent = ' Hide Details';
                                    row.querySelector('.toggle-details').prepend(row.querySelector('.toggle-details i'));
                                }
                            } else {
                                row.style.display = 'none';
                                detailsRow.style.display = 'none';
                            }
                        }
                    });
                    
                    // If search field is cleared, collapse all details
                    if (!searchTerm) {
                        const tbody = document.querySelector('#coordinated-groups tbody');
                        if (tbody) {
                            document.querySelectorAll('.group-details').forEach(r => {
                                r.style.display = 'none';
                            });
                            
                            document.querySelectorAll('.toggle-details i').forEach(icon => {
                                icon.className = 'bi bi-chevron-down';
                            });
                            
                            document.querySelectorAll('.toggle-details').forEach(button => {
                                button.textContent = ' View Details';
                                button.prepend(button.querySelector('i'));
                            });
                        }
                    }
                });
            }
        }
    } catch (error) {
        console.error('Error updating coordinated behavior:', error);
        
        const coordGroupsEl = document.getElementById('coordinated-groups');
        if (coordGroupsEl) {
            coordGroupsEl.innerHTML = '<div class="alert alert-danger"><i class="bi bi-exclamation-triangle"></i> Error loading coordinated behavior data. Please try again.</div>';
        }
    }
}

function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
}

function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
}

function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
}

// Data storytelling function - generates a narrative from visualizations
async function generateDataStory(query) {
    try {
        console.log('Generating data story for query:', query);
        // Collect data from various endpoints to build the story
        const [summaryData, timeseriesData, topContributors, topicsData, coordinatedData, wordsData] = await Promise.all([
            fetch(`/api/ai_summary?query=${encodeURIComponent(query)}`).then(r => r.json()).catch(() => null),
            fetch(`/api/timeseries?query=${encodeURIComponent(query)}`).then(r => r.json()).catch(() => []),
            fetch(`/api/top_contributors?query=${encodeURIComponent(query)}&limit=5`).then(r => r.json()).catch(() => []),
            fetch(`/api/topics?query=${encodeURIComponent(query)}&n_topics=3`).then(r => r.json()).catch(() => ({topics: []})),
            fetch(`/api/coordinated?query=${encodeURIComponent(query)}&time_window=3600&similarity_threshold=0.7`).then(r => r.json()).catch(() => null),
            fetch(`/api/common_words?query=${encodeURIComponent(query)}&limit=20`).then(r => r.json()).catch(() => [])
        ]);
        
        // Get the existing data story container
        const storyContainer = document.getElementById('data-story');
        if (!storyContainer) {
            console.error('Data story container not found');
            return;
        }
        
        // Extract key information for the case study
        let timeSpan = 'the analyzed period';
        let peakDate = 'recent days';
        let totalPosts = 0;
        let uniqueAuthors = 0;
        let mainCommunities = [];
        let topKeywords = [];
        let dominantTopic = '';
        let mainContributors = [];
        
        // Process time data
        if (timeseriesData && timeseriesData.length > 0) {
            // Sort by date
            timeseriesData.sort((a, b) => new Date(a.date) - new Date(b.date));
            
            // Calculate timespan
            const firstDate = new Date(timeseriesData[0].date);
            const lastDate = new Date(timeseriesData[timeseriesData.length - 1].date);
            const daysDiff = Math.round((lastDate - firstDate) / (1000 * 60 * 60 * 24));
            timeSpan = `${daysDiff} days (${firstDate.toLocaleDateString()} to ${lastDate.toLocaleDateString()})`;
            
            // Find peak day
            const peakDay = [...timeseriesData].sort((a, b) => b.count - a.count)[0];
            if (peakDay) {
                peakDate = new Date(peakDay.date).toLocaleDateString();
            }
        }
        
        // Get summary metrics
        if (summaryData && summaryData.metrics) {
            totalPosts = summaryData.metrics.total_posts;
            uniqueAuthors = summaryData.metrics.unique_authors;
            
            if (summaryData.metrics.top_subreddits) {
                mainCommunities = Object.keys(summaryData.metrics.top_subreddits).slice(0, 3);
            }
        }
        
        // Get keywords
        if (wordsData && wordsData.length > 0) {
            topKeywords = wordsData.slice(0, 10).map(w => w.word);
        }
        
        // Get dominant topic
        if (topicsData && topicsData.topics && topicsData.topics.length > 0) {
            dominantTopic = topicsData.topics[0].top_words.slice(0, 5).join(', ');
        }
        
        // Get main contributors
        if (topContributors && topContributors.length > 0) {
            mainContributors = topContributors.slice(0, 3).map(c => ({
                name: c.author,
                count: c.count
            }));
        }
        
        // Determine if there's coordinated activity
        let hasCoordination = false;
        let coordinationDetails = null;
        if (coordinatedData && coordinatedData.groups && coordinatedData.groups.length > 0) {
            hasCoordination = true;
            const significantGroup = coordinatedData.groups[0];
            coordinationDetails = {
                groupSize: significantGroup.size,
                uniqueAuthors: significantGroup.unique_authors,
                timeSpan: Math.round(significantGroup.time_span / 60), // convert to minutes
                examplePost: significantGroup.posts[0].title
            };
        }
        
        // Build the case study narrative based on collected data
        let storyHtml = `
            <div class="card-body data-story-content">
                <div class="story-header mb-4">
                    <h3 class="story-title mb-3">Case Study: The <span class="text-primary">${query}</span> Phenomenon</h3>
                    <p class="story-introduction">
                        This case study examines the social media conversation around <strong>"${query}"</strong>,
                        analyzing ${totalPosts} posts from ${uniqueAuthors} unique contributors over ${timeSpan}.
                    </p>
                </div>
                
                <div class="story-section mb-4">
                    <h4 class="section-title">Background & Context</h4>
                    <p>
                        The discourse around <strong>"${query}"</strong> primarily took place in
                        ${mainCommunities.length > 0 ? `the communities of r/${mainCommunities.join(', r/')}` : 'various online communities'},
                        with peak activity occurring on <strong>${peakDate}</strong>.
                    </p>
                    <p>
                        Analysis reveals several key themes dominating the conversation, centered around
                        <strong>${dominantTopic || 'various interconnected topics'}</strong>. These themes reflect broader
                        social patterns and concerns that have shaped public discussion on this subject.
                    </p>
                </div>
                
                <div class="story-section mb-4">
                    <h4 class="section-title">Key Findings</h4>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="finding-card p-3 mb-3 border rounded shadow-sm">
                                <h5 class="finding-title"><i class="bi bi-chat-quote"></i> Discourse Patterns</h5>
                                <p>
                                    The conversation around <strong>"${query}"</strong> shows distinctive linguistic patterns,
                                    with frequent use of terms like <strong>${topKeywords.slice(0, 5).join('</strong>, <strong>')}</strong>.
                                </p>
                                <p>
                                    These key terms indicate a focus on
                                    ${dominantTopic ? `<strong>${dominantTopic.split(',')[0]}</strong> and related concepts` : 'several interrelated concepts'},
                                    suggesting a complex discourse landscape.
                                </p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="finding-card p-3 mb-3 border rounded shadow-sm">
                                <h5 class="finding-title"><i class="bi bi-people"></i> Community Dynamics</h5>
                                <p>
                                    ${mainContributors.length > 0 ?
                                        `The conversation was significantly shaped by key voices including <strong>${mainContributors.map(c => c.name).join('</strong>, <strong>')}</strong>,
                                        who collectively contributed ${mainContributors.reduce((sum, c) => sum + c.count, 0)} posts.` :
                                        `The conversation featured a diverse array of voices with no single dominant contributor.`}
                                </p>
                                <p>
                                    ${hasCoordination ?
                                        `Evidence suggests coordinated posting behavior, with ${coordinationDetails.groupSize} similar posts
                                        from ${coordinationDetails.uniqueAuthors} users within just ${coordinationDetails.timeSpan} minutes.` :
                                        `The conversation appears to have evolved organically, with no evidence of coordinated posting behavior.`}
                                </p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="finding-card p-3 mb-3 border rounded shadow-sm">
                        <h5 class="finding-title"><i class="bi bi-graph-up"></i> Temporal Patterns</h5>
                        <p>
                            Analysis of posting frequency reveals ${timeseriesData && timeseriesData.length > 5 ?
                                `distinct patterns over the ${timeSpan} period, with notable fluctuations corresponding to external events.` :
                                `a relatively ${timeseriesData && timeseriesData.length > 0 && timeseriesData[0].count > timeseriesData[timeseriesData.length - 1].count ? 'decreasing' : 'steady'} trend over time.`}
                        </p>
                        <p>
                            The peak activity on <strong>${peakDate}</strong> coincided with
                            ${dominantTopic ? `intensified discussion about <strong>${dominantTopic.split(',')[0]}</strong>` : 'heightened community interest'},
                            suggesting a potential correlation with ${mainCommunities.length > 0 ? `activities in r/${mainCommunities[0]}` : 'broader online trends'}.
                        </p>
                    </div>
                </div>
                
                <div class="story-section mb-4">
                    <h4 class="section-title">In-Depth Analysis</h4>
                    <div class="analysis-card p-3 border rounded shadow-sm">
                        <h5 class="analysis-subtitle mb-3">The Emergence of Key Narratives</h5>
                        <p>
                            Our analysis identifies several interconnected narratives that have shaped the discourse around
                            <strong>"${query}"</strong>. These narratives reflect not just topical interests but deeper community
                            values and concerns:
                        </p>
                        
                        <ul class="narrative-list mt-3">
                            ${topicsData && topicsData.topics ? topicsData.topics.slice(0, 3).map((topic, i) => `
                                <li class="mb-3">
                                    <strong>Narrative ${i+1}: ${topic.top_words.slice(0, 3).join(' & ')}</strong>
                                    <p>
                                        This narrative centers around ${topic.top_words.slice(0, 5).join(', ')}, reflecting
                                        ${i === 0 ? 'the dominant discourse' : i === 1 ? 'a secondary but significant perspective' : 'an alternative viewpoint'}
                                        within the community. ${topic.representative_docs && topic.representative_docs.length > 0 ?
                                            `A representative example states: "${topic.representative_docs[0].title}"` : ''}
                                    </p>
                                </li>
                            `).join('') : `
                                <li class="mb-3">
                                    <strong>Primary Narrative: Community Consensus</strong>
                                    <p>
                                        The conversation shows a relatively united perspective on ${query}, with limited evidence of competing narratives.
                                        The discourse primarily revolves around ${topKeywords.slice(0, 3).join(', ')}.
                                    </p>
                                </li>
                                <li class="mb-3">
                                    <strong>Secondary Narrative: Practical Applications</strong>
                                    <p>
                                        A secondary thread focuses on practical aspects and real-world implications, particularly in relation to
                                        ${topKeywords.slice(3, 6).join(', ')}.
                                    </p>
                                </li>
                            `}
                        </ul>
                    </div>
                </div>
                
                <div class="story-section mb-4">
                    <h4 class="section-title">Implications & Conclusions</h4>
                    <p>
                        The analysis of conversations around <strong>"${query}"</strong> reveals several key insights
                        with broader implications:
                    </p>
                    
                    <div class="row mt-3">
                        <div class="col-md-6">
                            <div class="implication-card p-3 border-start border-4 border-primary bg-light mb-3">
                                <h5 class="implications-title"><i class="bi bi-lightbulb"></i> Community Insights</h5>
                                <ul>
                                    <li>
                                        ${mainCommunities.length > 1 ?
                                            `Cross-community engagement suggests broad interest spanning multiple domains.` :
                                            `Concentrated discussion in specific communities indicates specialized interest.`}
                                    </li>
                                    <li>
                                        ${uniqueAuthors > totalPosts/2 ?
                                            `High ratio of unique contributors to posts indicates broad but shallow engagement.` :
                                            `Lower ratio of unique contributors to posts suggests deeper engagement from a core community.`}
                                    </li>
                                </ul>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="implication-card p-3 border-start border-4 border-success bg-light mb-3">
                                <h5 class="implications-title"><i class="bi bi-bar-chart"></i> Trend Significance</h5>
                                <ul>
                                    <li>
                                        ${timeseriesData && timeseriesData.length > 0 && timeseriesData[0].count < timeseriesData[timeseriesData.length - 1].count ?
                                            `The growing conversation suggests increasing relevance and public interest.` :
                                            `The conversation pattern suggests an established rather than emerging topic.`}
                                    </li>
                                    <li>
                                        ${hasCoordination ?
                                            `Evidence of coordinated activity warrants attention to potential organized information campaigns.` :
                                            `The organic evolution of discussion indicates authentic public interest.`}
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <div class="conclusion-card p-3 mt-3 border rounded shadow-sm">
                        <h5 class="conclusion-title text-center mb-3">Final Assessment</h5>
                        <p>
                            The conversation around <strong>"${query}"</strong> represents a
                            ${uniqueAuthors > 100 ? 'significant' : 'notable'}
                            discourse phenomenon characterized by
                            ${hasCoordination ? 'elements of coordination alongside' : ''}
                            organic community engagement. The discourse is primarily
                            ${dominantTopic ? `centered around ${dominantTopic.split(',')[0]}` : 'multifaceted'},
                            with implications for understanding both online community dynamics and broader public interest in this subject.
                        </p>
                    </div>
                </div>
                
                
            </div>
        `;
        
        // Add the story to the container
        storyContainer.innerHTML = storyHtml;
        
        // Add custom CSS for the data story
        const styleElement = document.createElement('style');
        styleElement.textContent = `
            .data-story-content {
                font-family: 'Lora', serif;
                line-height: 1.6;
            }
            .story-title {
                font-weight: 700;
                color: #212529;
                margin-bottom: 1rem;
            }
            .section-title {
                font-weight: 600;
                color: #343a40;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #dee2e6;
            }
            .finding-title, .analysis-subtitle, .implications-title, .conclusion-title {
                font-weight: 600;
                color: #495057;
                margin-bottom: 0.8rem;
            }
            .finding-card, .analysis-card, .implication-card, .conclusion-card {
                border-radius: 0.5rem;
                transition: all 0.3s ease;
            }
            .finding-card:hover, .analysis-card:hover, .conclusion-card:hover {
                box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15) !important;
            }
            .narrative-list li {
                margin-bottom: 1rem;
            }
            .narrative-list li strong {
                color: #0d6efd;
            }
            .story-introduction {
                font-size: 1.1rem;
                color: #495057;
            }
            @media print {
                .dashboard-header, .controls, .nav-tabs, .container-fluid > *:not(#data-story) {
                    display: none !important;
                }
                .story-footer button {
                    display: none;
                }
                #data-story {
                    box-shadow: none !important;
                    border: none !important;
                }
                .data-story-content {
                    font-size: 12pt;
                }
                .section-title {
                    font-size: 16pt;
                }
                .finding-title, .analysis-subtitle, .implications-title, .conclusion-title {
                    font-size: 14pt;
                }
            }
        `;
        document.head.appendChild(styleElement);
        
        const desc = document.getElementById('data-story-description');
        if (desc) desc.style.display = 'none';
    } catch (error) {
        console.error('Error generating data story:', error);
        // Show error message if generation fails
        const storyContainer = document.getElementById('data-story');
        if (storyContainer) {
            storyContainer.innerHTML = `
                <div class="card-body">
                    <div class="alert alert-danger">
                        <i class="bi bi-exclamation-triangle"></i>
                        Error generating case study. Please try a different query or reload the page.
                    </div>
                </div>
            `;
        }
    }
}

// Add CSS for section loading indicators
document.addEventListener('DOMContentLoaded', function() {
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .section-loading {
            padding: 20px;
            text-align: center;
            color: #6c757d;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .section-loading .spinner-border {
            margin-right: 10px;
        }
    `;
    document.head.appendChild(styleElement);
});

// Initialize the dashboard after document loads
document.addEventListener('DOMContentLoaded', function() {
    // Hide loading indicator initially
    showLoading(false);
    
    // Set up event listeners
    document.getElementById('analyze-btn').addEventListener('click', handleAnalyzeClick);
    
    // Set up topics range slider
    const topicsCountSlider = document.getElementById('topics-count');
    if (topicsCountSlider) {
        topicsCountSlider.addEventListener('input', function() {
            document.getElementById('topics-count-value').textContent = this.value;
        });
        
        document.getElementById('update-topics-btn').addEventListener('click', function() {
            const topicsCount = topicsCountSlider.value;
            const query = document.getElementById('query-input').value;
            updateTopics(query, topicsCount);
        });
    }
    
    // Tab change event listener - render charts if empty
    const analyticsTabs = document.getElementById('analyticsTabs');
    if (analyticsTabs) {
        analyticsTabs.addEventListener('shown.bs.tab', async function(event) {
            const targetId = event.target.getAttribute('data-bs-target');
            const activeQuery = document.getElementById('query-input').value;
            
            if (!activeQuery) return;
            
            // ... existing code ...
        });
    }
    
    // Event listener for manual update of coordinated behavior (removed button)
    const timeWindowSlider = document.getElementById('time-window');
    const similarityThresholdSlider = document.getElementById('similarity-threshold');
    
    if (timeWindowSlider) {
        timeWindowSlider.addEventListener('change', async () => {
            await updateCoordinatedBehavior();
            // Update coordinated description again after refresh
            updateSectionDescription('coordinated', '#coordinated-description', {
                timeWindow: document.getElementById('time-window').value,
                similarityThreshold: document.getElementById('similarity-threshold').value
            });
        });
    }
    
    if (similarityThresholdSlider) {
        similarityThresholdSlider.addEventListener('change', async () => {
            await updateCoordinatedBehavior();
            // Update coordinated description again after refresh
            updateSectionDescription('coordinated', '#coordinated-description', {
                timeWindow: document.getElementById('time-window').value,
                similarityThreshold: document.getElementById('similarity-threshold').value
            });
        });
    }
});

// Function to set up network controls
function setupNetworkControls() {
    // Network type radio buttons
    const networkTypeRadios = document.querySelectorAll('input[name="network-type"]');
    const contentOptions = document.getElementById('content-options');
    const similaritySliderContainer = document.getElementById('similarity-slider-container');
    const minSimilaritySlider = document.getElementById('min-similarity-slider');
    const similarityValueDisplay = document.getElementById('similarity-value');
    const updateNetworkBtn = document.getElementById('update-network-btn');
    
    // Initially hide content-specific controls when in interaction mode
    if (contentOptions && similaritySliderContainer) {
        // Check initial state
        const isContentNetwork = document.getElementById('content-network').checked;
        contentOptions.style.display = isContentNetwork ? 'block' : 'none';
        similaritySliderContainer.style.display = isContentNetwork ? 'block' : 'none';
    }
    
    // Add event listeners to network type radio buttons
    if (networkTypeRadios) {
        networkTypeRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                if (contentOptions && similaritySliderContainer) {
                    const isContentNetwork = this.value === 'content_sharing';
                    contentOptions.style.display = isContentNetwork ? 'block' : 'none';
                    similaritySliderContainer.style.display = isContentNetwork ? 'block' : 'none';
                }
            });
        });
    }
    
    // Add event listener to similarity slider
    if (minSimilaritySlider && similarityValueDisplay) {
        minSimilaritySlider.addEventListener('input', function() {
            const sliderValue = parseFloat(this.value) * 100;
            similarityValueDisplay.textContent = sliderValue.toFixed(0) + '%';
        });
    }
    
    // Add event listener to update network button
    if (updateNetworkBtn) {
        updateNetworkBtn.addEventListener('click', async function() {
            const query = document.getElementById('query-input').value;
            if (query) {
                try {
                    await updateNetwork(query);
                    // Update network description after data is loaded
                    updateSectionDescription('network', '#network-description', {
                        nodeCount: document.querySelectorAll('#network-graph circle').length
                    });
                } catch (error) {
                    console.error('Error updating network:', error);
                    document.getElementById('network-graph').innerHTML = '<p class="text-danger">Error loading network data</p>';
                }
            } else {
                alert('Please enter a search query first');
            }
        });
    }
}

// Topic Evolution Visualization
async function updateTopicEvolution(query) {
    try {
        // First, get topic data
        const topicsCount = document.getElementById('topics-count').value || 5;
        const response = await fetch(`/api/topics?n_topics=${topicsCount}&query=${encodeURIComponent(query)}`);
        
        if (!response.ok) {
            throw new Error(`Topics request failed with status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Check if we have valid data with topic evolution
        if (!data || !data.topic_evolution || Object.keys(data.topic_evolution).length === 0) {
            document.getElementById('topic-evolution-chart').innerHTML = 
                '<div class="alert alert-info">Not enough data available to show topic evolution for this query.</div>';
            return;
        }
        
        // Clear previous chart
        d3.select('#topic-evolution-chart').html('');
        
        const chartContainer = document.getElementById('topic-evolution-chart');
        const containerWidth = chartContainer.clientWidth || 800;
        const margin = {top: 60, right: 120, bottom: 90, left: 80}; // Increased right margin to ensure label visibility
        const width = containerWidth - margin.left - margin.right;
        const height = 550 - margin.top - margin.bottom; // Further increased height for better spacing
        
        // Create responsive SVG container
        const svg = d3.select('#topic-evolution-chart')
            .append('svg')
            .attr('width', '100%') // Use 100% width to fill container
            .attr('height', height + margin.top + margin.bottom)
            .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Process data for visualization
        const timeData = {};
        const allDates = new Set();
        const topicIds = [];
        
        // Collect all dates and topic ids
        for (const [topicId, dateCounts] of Object.entries(data.topic_evolution)) {
            topicIds.push(topicId);
            for (const date of Object.keys(dateCounts)) {
                allDates.add(date);
            }
        }
        
        // Sort dates chronologically
        const sortedDates = Array.from(allDates).sort();
        
        // Create a mapping of topic IDs to human-readable topic labels
        const topicLabels = {};
        topicIds.forEach(topicId => {
            const topicNumber = topicId.split('_')[1];
            const topic = data.topics.find(t => t.topic_id == topicNumber);
            if (topic && topic.top_words) {
                topicLabels[topicId] = `Topic ${topicNumber}: ${topic.top_words.slice(0, 3).join(', ')}`;
            } else {
                topicLabels[topicId] = `Topic ${topicNumber}`;
            }
        });
        
        // Create data points for each topic over time
        const chartData = [];
        topicIds.forEach((topicId, i) => {
            const topicData = {
                id: topicId,
                name: topicLabels[topicId],
                values: []
            };
            
            // Add values for all dates (filling in zeros for missing dates)
            sortedDates.forEach(date => {
                const dateObj = new Date(date);
                const value = data.topic_evolution[topicId][date] || 0;
                topicData.values.push({
                    date: dateObj,
                    value: value,
                    topicId: topicId
                });
            });
            
            chartData.push(topicData);
        });
        
        // Smooth the data (simple moving average)
        const smoothingWindow = Math.min(3, sortedDates.length);
        chartData.forEach(topic => {
            if (smoothingWindow > 1) {
                for (let i = 0; i < topic.values.length; i++) {
                    let sum = 0;
                    let count = 0;
                    
                    for (let j = Math.max(0, i - Math.floor(smoothingWindow/2)); 
                         j <= Math.min(topic.values.length - 1, i + Math.floor(smoothingWindow/2)); j++) {
                        sum += topic.values[j].value;
                        count++;
                    }
                    
                    topic.values[i].smoothedValue = sum / count;
                }
            } else {
                topic.values.forEach(v => v.smoothedValue = v.value);
            }
        });
        
        // Find topics with significant trends (increase or decrease)
        chartData.forEach(topic => {
            if (topic.values.length > 2) {
                const firstHalf = topic.values.slice(0, Math.floor(topic.values.length/2));
                const secondHalf = topic.values.slice(Math.floor(topic.values.length/2));
                
                const firstHalfAvg = d3.mean(firstHalf, d => d.value) || 0;
                const secondHalfAvg = d3.mean(secondHalf, d => d.value) || 0;
                
                // Calculate trend change percentage
                if (firstHalfAvg > 0) {
                    topic.trendChange = ((secondHalfAvg - firstHalfAvg) / firstHalfAvg * 100).toFixed(1);
                } else {
                    topic.trendChange = secondHalfAvg > 0 ? "∞" : "0";
                }
                
                // Determine trend direction
                if (secondHalfAvg > firstHalfAvg * 1.2) {
                    topic.trend = 'rising';
                } else if (secondHalfAvg < firstHalfAvg * 0.8) {
                    topic.trend = 'falling';
                } else {
                    topic.trend = 'stable';
                }
            } else {
                topic.trend = 'unknown';
                topic.trendChange = 'N/A';
            }
        });
        
        // Sort topics by trend (rising first, then falling, then stable)
        // and limit to maximum 5 topics to prevent label overcrowding
        chartData.sort((a, b) => {
            const trendOrder = { 'rising': 0, 'stable': 1, 'falling': 2, 'unknown': 3 };
            return trendOrder[a.trend] - trendOrder[b.trend];
        });
        
        // Limit to max 5 most significant topics to prevent overcrowding
        if (chartData.length > 5) {
            // Keep only the most significant topics (those with highest average values)
            chartData = chartData.map(topic => {
                const avgValue = d3.mean(topic.values, d => d.value);
                return {...topic, avgValue};
            })
            .sort((a, b) => b.avgValue - a.avgValue)
            .slice(0, 5);
        }
        
        // Setup scales
        const x = d3.scaleTime()
            .domain(d3.extent(sortedDates, d => new Date(d)))
            .range([0, width]);
        
        // Find max value across all topics and dates
        const maxValue = d3.max(chartData, d => d3.max(d.values, v => v.value)) * 1.1;
        
        const y = d3.scaleLinear()
            .domain([0, maxValue])
            .range([height, 0]);
        
        // Color scale for topics
        const color = d3.scaleOrdinal()
            .domain(topicIds)
            .range(d3.schemeCategory10);
        
        // Add axes with improved formatting
        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .attr('class', 'x-axis')
            .call(d3.axisBottom(x)
                .ticks(Math.min(sortedDates.length, width > 600 ? 10 : 5))
                .tickFormat(d3.timeFormat('%b %d')))
            .selectAll('text')
            .style('text-anchor', 'end')
            .attr('dx', '-.8em')
            .attr('dy', '.15em')
            .attr('transform', 'rotate(-45)')
            .style('font-size', '11px');
        
        svg.append('g')
            .attr('class', 'y-axis')
            .call(d3.axisLeft(y)
                .ticks(8)
                .tickFormat(d => {
                    if (d >= 1000) return d3.format(',.1k')(d);
                    return d;
                }))
            .selectAll('text')
            .style('font-size', '11px');
        
        // Add title with improved positioning to prevent overlaps
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', -margin.top / 2 + 10)
            .attr('text-anchor', 'middle')
            .attr('fill', 'var(--text-primary)')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text(`Topic Evolution for "${query}"`);
        
        // Create area generator
        const area = d3.area()
            .x(d => x(d.date))
            .y0(height)
            .y1(d => y(d.smoothedValue))
            .curve(d3.curveMonotoneX);
        
        // Create line generator
        const line = d3.line()
            .x(d => x(d.date))
            .y(d => y(d.smoothedValue))
            .curve(d3.curveMonotoneX);
        
        // Add topic areas with reduced opacity
        chartData.forEach(topic => {
            svg.append('path')
                .datum(topic.values)
                .attr('fill', color(topic.id))
                .attr('fill-opacity', 0.1)
                .attr('d', area);
        });
        
        // Add topic lines
        chartData.forEach(topic => {
            svg.append('path')
                .datum(topic.values)
                .attr('fill', 'none')
                .attr('stroke', color(topic.id))
                .attr('stroke-width', 2)
                .attr('d', line);
            
            // Add circles at the last point of each line
            const lastPoint = topic.values[topic.values.length - 1];
            
            svg.append('circle')
                .attr('cx', x(lastPoint.date))
                .attr('cy', y(lastPoint.smoothedValue))
                .attr('r', 4)
                .attr('fill', color(topic.id))
                .attr('stroke', '#fff')
                .attr('stroke-width', 1);
        });
        
        // Add topic labels at the end of each line with significantly improved spacing
        chartData.forEach((topic, i) => {
            const lastPoint = topic.values[topic.values.length - 1];
            
            // Stagger the labels vertically with much more space to avoid overlap
            const yOffset = i * 35 - (chartData.length * 9) + 10;
            
            svg.append('line')
                .attr('x1', x(lastPoint.date))
                .attr('y1', y(lastPoint.smoothedValue))
                .attr('x2', width + 5)
                .attr('y2', y(lastPoint.smoothedValue) + yOffset)
                .attr('stroke', color(topic.id))
                .attr('stroke-width', 1)
                .attr('stroke-dasharray', '3,3')
                .attr('opacity', 0.7);
            
            // Create a group for the label, positioned closer to the chart edge
            const labelGroup = svg.append('g')
                .attr('transform', `translate(${width + 10}, ${y(lastPoint.smoothedValue) + yOffset})`);
            
            // Add trend indicator symbol
            const trendSymbol = topic.trend === 'rising' ? '↑' : (topic.trend === 'falling' ? '↓' : '→');
            const trendColor = topic.trend === 'rising' ? '#20c997' : (topic.trend === 'falling' ? '#dc3545' : '#6c757d');
            
            labelGroup.append('text')
                .attr('x', 0)
                .attr('y', 0)
                .style('font-size', '14px')
                .style('font-weight', 'bold')
                .style('fill', trendColor)
                .text(trendSymbol);
            
            // Add topic label text with improved handling for readability
            const topWords = topic.name.split(':')[1] || '';
            // Drastically limit topic words to ensure trend percentages remain visible
            const shortenedWords = topWords.length > 6 ? topWords.substring(0, 6) + '...' : topWords;
            
            // Add text with background for better readability
            const textLabel = labelGroup.append('text')
                .attr('x', 12)
                .attr('y', 0)
                .style('font-size', '11px')
                .style('font-weight', 'normal')
                .style('dominant-baseline', 'middle')
                .text(shortenedWords);
            
            // Add trend percentage with improved positioning - separate line from topic name
            if (topic.trend !== 'unknown') {
                const trendText = topic.trend === 'rising' ? `+${topic.trendChange}%` : 
                                 (topic.trend === 'falling' ? `${topic.trendChange}%` : '');
                
                if (trendText) {
                    // Position trend percentage directly after topic name
                    labelGroup.append('text')
                        .attr('x', 60)
                        .attr('y', 0)
                        .style('font-size', '10px')
                        .style('font-weight', 'bold')
                        .style('dominant-baseline', 'middle')
                        .style('fill', trendColor)
                        .text(trendText);
                }
            }
        });
        
        // Add X axis label with improved styling
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height + margin.bottom - 15)
            .style('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '500')
            .attr('fill', 'var(--text-primary)')
            .text('Date');
        
        // Add Y axis label with improved styling
        svg.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('y', 0 - margin.left + 15)
            .attr('x', 0 - (height / 2))
            .attr('dy', '1em')
            .style('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', '500')
            .attr('fill', 'var(--text-primary)')
            .text('Topic Prevalence');
        
        // Add interactive hover effects
        const focusLine = svg.append('line')
            .attr('class', 'focus-line')
            .attr('stroke', '#999')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '3,3')
            .attr('y1', 0)
            .attr('y2', height)
            .style('opacity', 0);
        
        const focusCircles = chartData.map(topic => {
            return svg.append('circle')
                .attr('class', 'focus-circle')
                .attr('r', 6)
                .attr('stroke', color(topic.id))
                .attr('stroke-width', 2)
                .attr('fill', '#fff')
                .style('opacity', 0);
        });
        
        const focusTextBg = svg.append('rect')
            .attr('class', 'focus-text-bg')
            .attr('fill', 'white')
            .attr('rx', 5)
            .attr('ry', 5)
            .attr('width', 180)
            .attr('height', chartData.length * 20 + 30)
            .style('opacity', 0);
        
        const focusText = svg.append('g')
            .attr('class', 'focus-text')
            .style('opacity', 0);
        
        // Add mouse tracking overlay
        const overlay = svg.append('rect')
            .attr('width', width)
            .attr('height', height)
            .style('fill', 'none')
            .style('pointer-events', 'all')
            .on('mouseover', function() {
                focusLine.style('opacity', 1);
                focusCircles.forEach(circle => circle.style('opacity', 1));
                focusTextBg.style('opacity', 0.9);
                focusText.style('opacity', 1);
            })
            .on('mouseout', function() {
                focusLine.style('opacity', 0);
                focusCircles.forEach(circle => circle.style('opacity', 0));
                focusTextBg.style('opacity', 0);
                focusText.style('opacity', 0);
            })
            .on('mousemove', mousemove);
        
        function mousemove(event) {
            const mouse = d3.pointer(event);
            const dateValue = x.invert(mouse[0]);
            
            // Find the closest date in our data
            let closestDate = sortedDates[0];
            let closestDistance = Math.abs(new Date(closestDate) - dateValue);
            
            for (let i = 1; i < sortedDates.length; i++) {
                const distance = Math.abs(new Date(sortedDates[i]) - dateValue);
                if (distance < closestDistance) {
                    closestDate = sortedDates[i];
                    closestDistance = distance;
                }
            }
            
            // Position the line at the closest date
            const xPos = x(new Date(closestDate));
            focusLine.attr('x1', xPos).attr('x2', xPos);
            
            // Set position for tooltip
            const tooltipX = xPos > width/2 ? xPos - 190 : xPos + 10;
            const tooltipY = 20;
            
            focusTextBg
                .attr('x', tooltipX)
                .attr('y', tooltipY);
            
            focusText.selectAll('*').remove();
            
            // Add date to tooltip
            focusText.append('text')
                .attr('x', tooltipX + 10)
                .attr('y', tooltipY + 20)
                .style('font-weight', 'bold')
                .text(new Date(closestDate).toLocaleDateString());
            
            // Update circle positions and add topic values to tooltip
            chartData.forEach((topic, i) => {
                const topicData = topic.values.find(v => v.date.getTime() === new Date(closestDate).getTime());
                
                if (topicData) {
                    // Position circle
                    focusCircles[i]
                        .attr('cx', xPos)
                        .attr('cy', y(topicData.smoothedValue));
                    
                    // Add topic data to tooltip
                    focusText.append('text')
                        .attr('x', tooltipX + 10)
                        .attr('y', tooltipY + 20 + ((i + 1) * 20))
                        .style('fill', color(topic.id))
                        .text(`${topic.name.split(':')[1]}: ${Math.round(topicData.value * 100) / 100}`);
                }
            });
        }
        
        // Update the topic evolution description after data is loaded
        updateSectionDescription('topic_evolution', '#topic-evolution-description', {
            topicCount: topicsCount,
            trendData: chartData.map(topic => ({
                name: topic.name,
                trend: topic.trend,
                change: topic.trendChange
            }))
        });
        
    } catch (error) {
        console.error('Error updating topic evolution:', error);
        document.getElementById('topic-evolution-chart').innerHTML = 
            `<div class="alert alert-danger">Error loading topic evolution data: ${error.message}</div>`;
    }
}

// Community Distribution Pie Chart
async function updateCommunityDistributionPieChart(query) {
    try {
        // Get top subreddits from API
        const response = await fetch(`/api/top_contributors?query=${encodeURIComponent(query)}&limit=15`);
        const data = await response.json();
        
        // Format data for pie chart
        let subredditData = [];
        
        if (data && data.length > 0) {
            // If we have author data, try to process it as subreddit data
            subredditData = data.map(item => ({
                name: item.author.includes('/') ? item.author : item.author.trim(),
                count: item.count
            }));
        } else {
            // Fallback to empty data
            document.getElementById('community-distribution').innerHTML =
                '<div class="alert alert-warning">No community data available for this query</div>';
            return;
        }
        
        // Only keep top 10 communities, group others into "Other"
        subredditData.sort((a, b) => b.count - a.count);
        
        // Clear previous chart
        d3.select('#community-distribution').html('');
        
        const container = document.getElementById('community-distribution');
        const width = container.clientWidth || 600;
        const height = Math.min(500, width * 0.8);
        const radius = Math.min(width, height) / 2 - 40; // Increased margin for labels
        
        // Create SVG with adjusted positioning
        const svg = d3.select('#community-distribution')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${width / 2 + 30},${height / 2})`);
        
        // Calculate total posts for percentage display
        const totalPosts = subredditData.reduce((sum, item) => sum + item.count, 0);
        
        // Group smaller communities into "Other" if they are less than 2% of total
        const threshold = totalPosts * 0.02;
        let processedData = [];
        let otherCount = 0;
        
        subredditData.forEach(item => {
            if (item.count >= threshold) {
                processedData.push(item);
            } else {
                otherCount += item.count;
            }
        });
        
        // Add "Other" category if we have grouped communities
        if (otherCount > 0) {
            processedData.push({
                name: "Other",
                count: otherCount
            });
        }
        
        // Set color scale using dashboard theme colors
        const themeColors = [
            '#4a6fa5', // --primary-color
            '#6b9080', // --secondary-color
            '#ee6c4d', // --accent-color
            '#2c4674', // --primary-dark
            '#d5e3f6', // --primary-light
            '#e3f0ea', // --secondary-light
            '#f9e5dc', // --accent-light
            '#293241', // --dark-color
            '#6c757d', // --text-secondary
            '#93a8c7', // Custom shade
            '#8bb09f', // Custom shade
            '#f28e73'  // Custom shade
        ];
        
        const color = d3.scaleOrdinal()
            .domain(processedData.map(d => d.name))
            .range(themeColors);
        
        // Compute the position of each group on the pie
        const pie = d3.pie()
            .value(d => d.count)
            .sort(null); // Keep the original order
        
        const pieData = pie(processedData);
        
        // Shape helper to build arcs
        const arc = d3.arc()
            .innerRadius(radius * 0.5) // Slightly larger inner radius for more modern look
            .outerRadius(radius)
            .cornerRadius(3); // Slightly rounded corners for a modern look
        
        // Arc for hover effect
        const hoverArc = d3.arc()
            .innerRadius(radius * 0.48) // Slightly larger inner radius for hover effect
            .outerRadius(radius * 1.03)
            .cornerRadius(3);
            
        // Another arc for labels (not used for rendering labels, but kept for calculations)
        const outerArc = d3.arc()
            .innerRadius(radius * 1.1)
            .outerRadius(radius * 1.1);
        
        // Create a group for the donut chart
        const donutGroup = svg.append('g');
        
        // Build the pie chart with animations
        const slices = donutGroup.selectAll('path')
            .data(pieData)
            .enter()
            .append('path')
            .attr('d', arc)
            .attr('fill', d => color(d.data.name))
            .attr('stroke', 'white')
            .style('stroke-width', '1.5px')
            .style('opacity', 0.9)
            .style('transition', 'opacity 0.3s, filter 0.3s')
            // Add entrance animation
            .style('opacity', 0)
            .transition()
            .duration(800)
            .delay((d, i) => i * 50)
            .style('opacity', 0.9);
            
        // Add hover effects after the transition
        donutGroup.selectAll('path')
            .on('mouseover', function(event, d) {
                // Highlight the hovered slice
                d3.select(this)
                    .style('opacity', 1)
                    .style('filter', 'drop-shadow(0px 3px 5px rgba(0,0,0,0.2))')
                    .transition()
                    .duration(200)
                    .attr('d', hoverArc);
                
                // Dim other slices
                donutGroup.selectAll('path')
                    .filter(path => path !== d)
                    .style('opacity', 0.6);
                
                // Update tooltip with detailed information
                tooltip.transition()
                    .duration(200)
                    .style('opacity', 0.95);
                    
                const percentage = Math.round((d.data.count / totalPosts) * 100);
                tooltip.html(`
                    <div style="font-weight: bold; margin-bottom: 5px;">${d.data.name}</div>
                    <div>Posts: ${d.data.count}</div>
                    <div>Percentage: ${percentage}%</div>
                `)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
                
                // Show percentage in center
                centerText.text(`${d.data.name}`);
                centerSubText.text(`${d.data.count} posts (${percentage}%)`);
                centerTextGroup.style('opacity', 1);
            })
            .on('mouseout', function() {
                // Restore the slice to normal
                d3.select(this)
                    .style('opacity', 0.9)
                    .style('filter', 'none')
                    .transition()
                    .duration(200)
                    .attr('d', arc);
                
                // Restore other slices
                donutGroup.selectAll('path')
                    .style('opacity', 0.9);
                
                // Hide tooltip
                tooltip.transition()
                    .duration(500)
                    .style('opacity', 0);
                
                // Reset center text
                centerText.text('Community');
                centerSubText.text('Distribution');
            });
            
        // Create a group for the center text
        const centerTextGroup = svg.append('g')
            .style('opacity', 0.7)
            .on('mouseover', function() {
                centerTextGroup.style('opacity', 1);
            })
            .on('mouseout', function() {
                centerTextGroup.style('opacity', 0.7);
            });
            
        // Add hoverable center text
        const centerText = centerTextGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0em')
            .style('font-size', '1.3em')
            .style('font-weight', 'bold')
            .style('fill', 'var(--primary-dark)')
            .text('Community');
        
        const centerSubText = centerTextGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '1.5em')
            .style('font-size', '1em')
            .style('fill', 'var(--text-secondary)')
            .text('Distribution');
            
        // Add tooltip div for hover details
        const tooltip = d3.select('#community-distribution')
            .append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0)
            .style('position', 'absolute')
            .style('background-color', 'white')
            .style('border', '1px solid #ddd')
            .style('border-radius', '5px')
            .style('padding', '10px')
            .style('box-shadow', '0 2px 4px rgba(0,0,0,0.1)')
            .style('pointer-events', 'none')
            .style('font-size', '14px')
            .style('z-index', 1000);
            
        // Helper function to compute the angle in the middle of an arc
        function midAngle(d) {
            return d.startAngle + (d.endAngle - d.startAngle) / 2;
        }
            
        // Create side legend with scrollable container and improved positioning to prevent overlap
        const legendContainer = svg.append('foreignObject')
            .attr('x', -width/2 + 10)
            .attr('y', -height/2 + 40)
            .attr('width', width/3.5) // Wider container for the legend
            .attr('height', height - 80)
            .append('xhtml:div')
            .style('height', '100%')
            .style('overflow-y', 'auto')
            .style('padding-right', '10px')
            .style('padding-bottom', '5px')
            .style('background-color', 'rgba(255,255,255,0.5)') // Semi-transparent background
            .style('border-radius', '5px');
            
        const legendHTML = processedData.map((item, i) => {
            const percent = Math.round(item.count / totalPosts * 100);
            return `
                <div style="display: flex; align-items: center; margin-bottom: 10px; font-size: 12px; opacity: 0; animation: fadeIn 0.3s forwards ${i * 100 + 600}ms;">
                    <div style="width: 12px; height: 12px; background-color: ${color(item.name)}; margin-right: 8px; border-radius: 2px;"></div>
                    <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90%;">
                        ${item.name} (${percent}%)
                    </div>
                </div>
            `;
        }).join('');
        
        legendContainer.html(`
            <style>
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(5px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            </style>
            <div style="font-weight: bold; margin-bottom: 12px; font-size: 14px; color: var(--primary-dark);">Communities:</div>
            <div style="border-left: 2px solid #eee; padding-left: 10px; margin-left: 5px;">
                ${legendHTML}
            </div>
        `);
        
        // Update the community distribution description
        updateSectionDescription('community_distribution', '#community-distribution-description', {
            communityCount: subredditData.length,
            topCommunities: subredditData.slice(0, 3).map(c => c.name),
            totalPosts: totalPosts,
            dominantCommunity: processedData[0].name,
            dominantPercentage: Math.round((processedData[0].count / totalPosts) * 100),
            communityDiversity: processedData.length,
            smallestCommunity: processedData.length > 1 ? 
                (processedData[processedData.length-1].name !== 'Other' ? 
                    processedData[processedData.length-1].name : 
                    processedData[processedData.length-2].name) : 
                'None',
            smallestPercentage: processedData.length > 1 ? 
                (processedData[processedData.length-1].name !== 'Other' ? 
                    Math.round((processedData[processedData.length-1].count / totalPosts) * 100) : 
                    Math.round((processedData[processedData.length-2].count / totalPosts) * 100)) : 
                0,
            otherPercentage: otherCount > 0 ? Math.round((otherCount / totalPosts) * 100) : 0,
            query: query
        });
        
    } catch (error) {
        console.error('Error updating community distribution:', error);
        document.getElementById('community-distribution').innerHTML = 
            `<div class="alert alert-danger">Error loading community distribution: ${error.message}</div>`;
    }
}

// Add updateTopicEvolution to the main analysis workflow
async function handleAnalyzeClick() {
    const query = document.getElementById('query-input').value;
    if (!query) {
        alert('Please enter a search query');
        return;
    }

    activeQuery = query;
    startDate = '';
    endDate = '';
    
    showLoading(true);
    
    try {
        // Clear all previous visualizations first - with null checks
        const clearElement = (id, defaultContent = '') => {
            const element = document.getElementById(id);
            if (element) {
                element.innerHTML = defaultContent;
            }
        };
        
        clearElement('topics-container');
        clearElement('contributors-overview');
        // Remove reference to network-graph which is no longer needed
        clearElement('coordinated-groups');
        clearElement('word-cloud');
        clearElement('timeseries-chart');
        clearElement('ai-summary'); // This element might not exist
        clearElement('topic-evolution-chart');
        clearElement('semantic-map-container');
        clearElement('point-details', '<p class="text-muted">Click on a point to see details</p>');
        clearElement('topic-clusters', '<p class="text-muted">Loading topic clusters...</p>');
        clearElement('community-distribution');
        
        // Reset data story with placeholder
        clearElement('data-story', `
            <div class="card-body">
                <p class="text-muted">Enter a search query to generate a data story or case study.</p>
            </div>
        `);
        
        // PERFORMANCE OPTIMIZATION: Load data progressively in phases
        // Phase 1: First load critical data for the overview tab
        const criticalPromises = [
            // Update key metrics - essential for overview
            updateKeyMetrics(query).catch(error => {
                console.error('Error updating key metrics:', error);
                // Error state is handled within updateKeyMetrics
            }),
            // Update top contributors (small visualization) - essential for overview
            updateContributorsOverview(query).catch(error => {
                console.error('Error updating contributors overview:', error);
                clearElement('contributors-overview', '<p class="text-danger">Error loading contributors data</p>');
            }),
            // Update word cloud - lightweight visualization for overview
            updateWordCloud(query).catch(error => {
                console.error('Error updating word cloud:', error);
                clearElement('word-cloud', '<p class="text-danger">Error loading word cloud data</p>');
            }),
        ];

        // Wait for critical components first
        await Promise.allSettled(criticalPromises);
        
        // Description updates have been disabled
        // No descriptions to update
        
        // Hide the main loading spinner as critical content is loaded
        showLoading(false);
        
        // Mark that analysis has been performed
        analysisPerformed = true;
        
        // PERFORMANCE OPTIMIZATION: Create placeholder loading indicators for remaining components
        clearElement('timeseries-chart', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading time series analysis...</div>');
        clearElement('topic-evolution-chart', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading topic evolution analysis...</div>');
        // Remove reference to network-graph which no longer exists
        clearElement('topics-container', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading topic analysis...</div>');
        clearElement('coordinated-groups', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading coordinated behavior analysis...</div>');
        clearElement('community-distribution', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading community distribution...</div>');
        clearElement('semantic-map-container', '<div class="section-loading"><div class="spinner-border spinner-border-sm text-primary" role="status"></div> Loading semantic map analysis...</div>');
        
        // Phase 2: Load the data story and time series (medium weight)
        // Generate a data story based on the analyzed results
        setTimeout(async () => {
            try {
                await generateDataStory(query);
                
                // Update time series (important for overview)
                await updateTimeSeries(query).catch(error => {
                    console.error('Error updating time series:', error);
                    clearElement('timeseries-chart', '<p class="text-danger">Error loading time series data</p>');
                });
                
                // Update topic evolution chart
                await updateTopicEvolution(query).catch(error => {
                    console.error('Error updating topic evolution:', error);
                    document.getElementById('topic-evolution-chart').innerHTML = '<p class="text-danger">Error loading topic evolution data</p>';
                });
                
                // Update semantic map
                await updateSemanticMap(query).catch(error => {
                    console.error('Error updating semantic map:', error);
                    document.getElementById('semantic-map-container').innerHTML = '<p class="text-danger">Error loading semantic map data</p>';
                });
                
                // Description updates have been disabled
                
                // Phase 3: Now load the remaining heavier visualizations in the background
                // This allows the user to start interacting with the dashboard while heavy visualizations load
                setTimeout(async () => {
                    // Process remaining heavy visualizations in sequence to reduce load
                    try {
                        await updateTopics(query);
                        // Description updates have been disabled
                    } catch (error) {
                        console.error('Error updating topics:', error);
                        clearElement('topics-container', '<p class="text-danger">Error loading topics data</p>');
                    }
                    
                    try {
                        // Network Analysis section has been removed
                        // Skip network update since that section no longer exists
                        // await updateNetwork(query);
                        // Update network description after data is loaded
                        // updateSectionDescription('network', '#network-description', {
                        //     nodeCount: document.querySelectorAll('#network-graph circle').length
                        // });
                    } catch (error) {
                        console.error('Error updating network:', error);
                        // Removed reference to network-graph which no longer exists
                        // document.getElementById('network-graph').innerHTML = '<p class="text-danger">Error loading network data</p>';
                    }
                    
                    try {
                        await updateCoordinatedBehavior();
                        // Description updates have been disabled
                    } catch (error) {
                        console.error('Error updating coordinated behavior:', error);
                        // Remove reference to coordinated-graph which no longer exists
                        clearElement('coordinated-groups', '<p class="text-danger">Error loading coordinated behavior data</p>');
                    }
                    
                    // Update pie chart of contributors
                    try {
                        await updateCommunityDistributionPieChart(query);
                    } catch (error) {
                        console.error('Error updating community distribution:', error);
                        clearElement('community-distribution', '<p class="text-danger">Error loading community distribution data</p>');
                    }
                    
                    console.log('All visualizations loaded');
                }, 100);
                
            } catch (error) {
                console.error('Error in phase 2 loading:', error);
            }
        }, 10);
        
    } catch (error) {
        console.error('Error during analysis:', error);
        alert('Error performing analysis. Please check the console for details.');
        showLoading(false);
    }
}

// Function to update the semantic map
async function updateSemanticMap(query) {
    try {
        // Show loading indicator
        document.getElementById('semantic-map-container').innerHTML = `
            <div class="section-loading">
                <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
                Loading semantic map...
            </div>
        `;
        document.getElementById('topic-clusters').innerHTML = `
            <div class="section-loading">
                <div class="spinner-border spinner-border-sm text-primary" role="status"></div>
                Loading topic clusters...
            </div>
        `;
        
        // Get parameters from UI controls
        const maxPoints = document.getElementById('max-points-slider').value;
        const nNeighbors = document.getElementById('n-neighbors-slider').value;
        const minDist = document.getElementById('min-dist-slider').value;
        
        // Build the URL with parameters
        let url = `/api/semantic_map?query=${encodeURIComponent(query)}&max_points=${maxPoints}&n_neighbors=${nNeighbors}&min_dist=${minDist}`;
        
        // Fetch semantic map data
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Check if we have valid data
        if (!data.points || data.points.length === 0) {
            document.getElementById('semantic-map-container').innerHTML = `
                <div class="alert alert-warning">
                    No data points available for the current query.
                </div>
            `;
            return;
        }
        
        // Update stats display
        document.getElementById('semantic-map-stats').innerHTML = `
            <strong>${data.total_posts}</strong> posts visualized | 
            <strong>${data.topics.length}</strong> topic clusters 
            
        `;
        
        // Visualize the data
        renderSemanticMap(data);
        
        // Update the clusters sidebar
        updateTopicClusters(data.topics);
        
        // Update section description
        updateSectionDescription('semantic_map', '#semantic-map-description', {
            pointCount: data.total_posts,
            topicCount: data.topics.length,
            parameters: data.umap_params
        });
        
    } catch (error) {
        console.error('Error updating semantic map:', error);
        document.getElementById('semantic-map-container').innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error loading semantic map: ${error.message}
            </div>
        `;
    }
}

// Function to render the semantic map visualization
function renderSemanticMap(data) {
    // Clear previous visualization
    document.getElementById('semantic-map-container').innerHTML = '';
    
    // Set up dimensions
    const container = document.getElementById('semantic-map-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = {top: 10, right: 30, bottom: 30, left: 40};
    
    // Create SVG
    const svg = d3.select('#semantic-map-container').append('svg')
        .attr('width', width)
        .attr('height', height);
        
    // Create a group for zoom/pan transform
    const g = svg.append('g');
    
    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 8])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    
    svg.call(zoom);
    
    // Compute the data boundaries
    const xExtent = d3.extent(data.points, d => d.x);
    const yExtent = d3.extent(data.points, d => d.y);
    
    // Add some padding to the extents
    const xPadding = (xExtent[1] - xExtent[0]) * 0.05;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.05;
    
    // Create scales
    const xScale = d3.scaleLinear()
        .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
        .range([margin.left, width - margin.right]);
        
    const yScale = d3.scaleLinear()
        .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
        .range([height - margin.bottom, margin.top]);
    
    // Create a color scale for clusters
    const showClusters = document.getElementById('show-clusters').checked;
    const clusterColorScale = d3.scaleOrdinal(d3.schemeCategory10)
        .domain(data.topics.map(t => t.id));
    
    // Create points
    g.selectAll('circle')
        .data(data.points)
        .enter()
        .append('circle')
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('r', 5)
        .attr('fill', d => showClusters ? clusterColorScale(d.cluster) : '#4a6fa5')
        .attr('opacity', 0.7)
        .attr('stroke', '#fff')
        .attr('stroke-width', 0.5)
        .attr('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            // Highlight point
            d3.select(this)
                .attr('r', 8)
                .attr('stroke-width', 1.5);
                
            // Show tooltip
            const tooltip = d3.select('#semantic-map-container')
                .append('div')
                .attr('class', 'semantic-tooltip')
                .style('position', 'absolute')
                .style('background', 'white')
                .style('border', '1px solid #ddd')
                .style('border-radius', '4px')
                .style('padding', '10px')
                .style('box-shadow', '0 2px 5px rgba(0,0,0,0.1)')
                .style('pointer-events', 'none')
                .style('left', `${event.pageX + 10}px`)
                .style('top', `${event.pageY + 10}px`)
                .style('z-index', 1000);
                
            tooltip.html(`
                <strong>${d.title}</strong><br>
                <span class="small text-muted">r/${d.subreddit} - ${d.author}</span><br>
                <span class="small">${new Date(d.created_utc).toLocaleDateString()}</span>
            `);
        })
        .on('mouseout', function() {
            // Reset point size
            d3.select(this)
                .attr('r', 5)
                .attr('stroke-width', 0.5);
                
            // Remove tooltip
            d3.select('.semantic-tooltip').remove();
        })
        .on('click', (event, d) => {
            showPointDetails(d);
        });
    
    // Show cluster topic labels if enabled
    if (document.getElementById('show-topic-labels').checked) {
        g.selectAll('text')
            .data(data.topics)
            .enter()
            .append('text')
            .attr('x', d => xScale(d.center_x))
            .attr('y', d => yScale(d.center_y))
            .attr('text-anchor', 'middle')
            .attr('font-size', d => Math.min(14, 10 + Math.sqrt(d.size/10)))
            .attr('font-weight', 'bold')
            .attr('pointer-events', 'none')
            .attr('fill', d => showClusters ? clusterColorScale(d.id) : '#333')
            .text(d => d.terms.slice(-3).join(' | '));
    }
    
    // Draw convex hulls around clusters if enabled
    if (document.getElementById('show-hulls').checked && showClusters) {
        // Group points by cluster
        const clusterGroups = {};
        data.points.forEach(p => {
            if (!clusterGroups[p.cluster]) {
                clusterGroups[p.cluster] = [];
            }
            clusterGroups[p.cluster].push([xScale(p.x), yScale(p.y)]);
        });
        
        // Draw hulls for each cluster
        Object.entries(clusterGroups).forEach(([clusterId, points]) => {
            if (points.length < 3) return; // Need at least 3 points for a hull
            
            try {
                // Compute hull points
                const hull = d3.polygonHull(points);
                
                if (!hull) return;
                
                // Create hull path
                const hullPath = 'M' + hull.join('L') + 'Z';
                
                // Add hull to the visualization
                g.append('path')
                    .attr('d', hullPath)
                    .attr('fill', clusterColorScale(clusterId))
                    .attr('stroke', clusterColorScale(clusterId))
                    .attr('stroke-width', 1)
                    .attr('fill-opacity', 0.1)
                    .attr('stroke-opacity', 0.3);
            } catch (e) {
                console.warn('Could not create hull for cluster', clusterId, e);
            }
        });
    }
    
    // Add initial zoom to fit all points
    svg.call(zoom.transform, d3.zoomIdentity
        .translate(width/2, height/2)
        .scale(0.85)
        .translate(-width/2, -height/2));
}

// Function to update the topic clusters sidebar
function updateTopicClusters(topics) {
    // Clear existing content
    const clusterContainer = document.getElementById('topic-clusters');
    clusterContainer.innerHTML = '';
    
    if (!topics || topics.length === 0) {
        clusterContainer.innerHTML = `<p class="text-muted">No topic clusters identified.</p>`;
        return;
    }
    
    // Sort topics by size (descending)
    const sortedTopics = [...topics].sort((a, b) => b.size - a.size);
    
    // Create a list of topic clusters
    const topicList = document.createElement('div');
    topicList.className = 'list-group';
    
    sortedTopics.forEach(topic => {
        const topicItem = document.createElement('div');
        topicItem.className = 'list-group-item list-group-item-action';
        
        const badge = `<span class="badge bg-primary float-end">${topic.size}</span>`;
        const terms = topic.terms.join(', ');
        
        topicItem.innerHTML = `
            <div class="d-flex w-100 justify-content-between">
                <h6 class="mb-1">Cluster ${topic.id + 1}</h6>
                ${badge}
            </div>
            <p class="mb-1 small">${terms}</p>
        `;
        
        topicList.appendChild(topicItem);
    });
    
    clusterContainer.appendChild(topicList);
}

// Function to show detailed information about a clicked point
function showPointDetails(point) {
    const detailsContainer = document.getElementById('point-details');
    
    const date = new Date(point.created_utc).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
    
    detailsContainer.innerHTML = `
        <h6>${point.title}</h6>
        <div class="mb-2">
            <span class="badge bg-secondary">r/${point.subreddit}</span>
            <span class="badge bg-light text-dark">u/${point.author}</span>
            <span class="badge bg-light text-dark">${date}</span>
        </div>
        <p class="small">${point.preview_text}</p>
        <div class="d-flex">
            <div class="me-3">
                <i class="bi bi-chat-dots"></i> ${point.num_comments} comments
            </div>
            <div>
                <i class="bi bi-arrow-up-circle"></i> ${point.score} score
            </div>
        </div>
    `;
}

// Set up event handlers for semantic map controls
document.addEventListener('DOMContentLoaded', function() {
    // Set up analyze button click handler
    document.getElementById('analyze-btn').addEventListener('click', handleAnalyzeClick);
    
    // Set up semantic map update button handler
    const updateBtn = document.getElementById('update-semantic-map-btn');
    if (updateBtn) {
        updateBtn.addEventListener('click', function() {
            if (!activeQuery) {
                alert('Please run a query first');
                return;
            }
            updateSemanticMap(activeQuery);
        });
    }
    
    // Update parameter value displays
    const maxPointsSlider = document.getElementById('max-points-slider');
    const nNeighborsSlider = document.getElementById('n-neighbors-slider');
    const minDistSlider = document.getElementById('min-dist-slider');
    
    if (maxPointsSlider) {
        maxPointsSlider.addEventListener('input', function() {
            document.getElementById('max-points-value').textContent = this.value;
        });
    }
    
    if (nNeighborsSlider) {
        nNeighborsSlider.addEventListener('input', function() {
            document.getElementById('n-neighbors-value').textContent = this.value;
        });
    }
    
    if (minDistSlider) {
        minDistSlider.addEventListener('input', function() {
            document.getElementById('min-dist-value').textContent = this.value;
        });
    }
    
    // Real-time display toggles
    const displayOptions = ['show-topic-labels', 'show-clusters', 'show-hulls'];
    displayOptions.forEach(id => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            checkbox.addEventListener('change', function() {
                // Only update if we have already performed analysis
                if (analysisPerformed && activeQuery) {
                    updateSemanticMap(activeQuery);
                }
            });
        }
    });
    
    // Set up chatbot interactions
    setupChatbotInteractions();
});

// Chatbot functionality
let chatHistory = [];

function setupChatbotInteractions() {
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-chat-btn');
    
    // Add welcome message
    addMessageToChat('system', 
        `<div class="welcome-message">
            <h4 class="mb-3">👋 Welcome to the AI Data Analysis Assistant!</h4>
            <p class="mb-2">I can help you analyze social media data and provide insights about:</p>
            <ul class="list-unstyled mb-3">
                <li class="mb-2">• Trending topics and discussions</li>
                <li class="mb-2">• Sentiment patterns and changes</li>
                <li class="mb-2">• User behavior and influence</li>
                <li class="mb-2">• Community interactions and coordination</li>
            </ul>
            <p class="mb-0">Try asking a question or click one of the suggested queries below!</p>
        </div>`
    );
    
    // Send message on button click
    if (sendButton) {
        sendButton.addEventListener('click', function() {
            sendChatMessage();
        });
    }
    
    // Send message on Enter key
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
}

// Function to handle suggested query clicks
function suggestQuery(button) {
    const chatInput = document.getElementById('chat-input');
    chatInput.value = button.textContent.trim();
    sendChatMessage();
}

function sendChatMessage() {
    const chatInput = document.getElementById('chat-input');
    const userQuery = chatInput.value.trim();
    
    if (!userQuery) return; // Don't send empty messages
    
    // Clear input field
    chatInput.value = '';
    
    // Add user message to chat
    addMessageToChat('user', userQuery);
    
    // Show typing indicator
    showTypingIndicator();
    
    // Add message to history
    chatHistory.push({ role: 'user', content: userQuery });
    
    // Send message to API
    processChatMessage(userQuery);
}

function addMessageToChat(role, content, time = null) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    
    // Set class based on message role
    messageDiv.className = role === 'user' ? 'user-message' : 
                          (role === 'system' ? 'system-message' : 'assistant-message');
    
    // Create message content
    const contentP = document.createElement('div');
    contentP.className = 'message-content';
    
    // Handle content based on role
    if (role === 'system' || role === 'assistant') {
        // For system and assistant messages, allow HTML but sanitize
        let cleanContent = content;
        
        // Clean up obvious code blocks or quote markers
        if (typeof cleanContent === 'string') {
            cleanContent = cleanContent.trim()
                .replace(/^```html\s*/g, '')
                .replace(/```\s*$/g, '')
                .replace(/^["'`]|["'`]$/g, '');
            
            // If content still doesn't look like HTML, wrap it
            if (!cleanContent.startsWith('<')) {
                cleanContent = `<p>${cleanContent}</p>`;
            }
        }
        
        // Set the HTML content
        contentP.innerHTML = cleanContent;
    } else {
        // For user messages, escape HTML
        contentP.textContent = content;
    }
    
    messageDiv.appendChild(contentP);
    
    // Add timestamp if provided
    if (time) {
        const timeSpan = document.createElement('div');
        timeSpan.className = 'message-time';
        timeSpan.textContent = time;
        messageDiv.appendChild(timeSpan);
    }
    
    // Add message to chat
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'typing-indicator';
    indicatorDiv.id = 'typing-indicator';
    
    // Add three dots
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'dot';
        indicatorDiv.appendChild(dot);
    }
    
    chatMessages.appendChild(indicatorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

function processChatMessage(query) {
    // Hide previous suggestions and metrics
    const vizSuggestions = document.getElementById('viz-suggestions');
    const relatedMetrics = document.getElementById('related-metrics');
    
    if (vizSuggestions) vizSuggestions.style.display = 'none';
    if (relatedMetrics) relatedMetrics.style.display = 'none';
    
    // Call API to process the message
    fetch('/api/chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: query,
            history: chatHistory.slice(-5) // Send last 5 messages for context
        })
    })
    .then(response => {
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        return response.json();
    })
    .then(data => {
        hideTypingIndicator();
        
        if (!data.response) throw new Error("Empty response from server");
        
        // Clean and format the response
        let cleanedResponse = data.response.trim()
            .replace(/^```html\s*/g, '')
            .replace(/```\s*$/g, '')
            .replace(/^["'`]|["'`]$/g, '')
            .replace(/^<div class=['"']chatbot-response['"']>|<\/div>$/g, '');
            
        // Add response to chat
        addMessageToChat('assistant', cleanedResponse);
        chatHistory.push({ role: 'assistant', content: cleanedResponse });
        
        // Add interactive elements to the response
        const responseElement = document.querySelector('.assistant-message:last-child .message-content');
        if (responseElement) {
            // Add copy buttons for code blocks
            responseElement.querySelectorAll('pre code').forEach(block => {
                const copyButton = document.createElement('button');
                copyButton.className = 'copy-code-btn';
                copyButton.innerHTML = '<i class="bi bi-clipboard"></i>';
                copyButton.onclick = () => {
                    navigator.clipboard.writeText(block.textContent);
                    copyButton.innerHTML = '<i class="bi bi-clipboard-check"></i>';
                    setTimeout(() => copyButton.innerHTML = '<i class="bi bi-clipboard"></i>', 2000);
                };
                block.parentNode.insertBefore(copyButton, block);
            });

            // Add collapsible sections
            responseElement.querySelectorAll('h3, h4').forEach(heading => {
                if (!heading.hasAttribute('data-collapsible')) {
                    heading.setAttribute('data-collapsible', 'true');
                    heading.style.cursor = 'pointer';
                    heading.innerHTML += ' <i class="bi bi-chevron-down"></i>';
                    heading.onclick = () => {
                        const content = heading.nextElementSibling;
                        if (content) {
                            const isVisible = content.style.display !== 'none';
                            content.style.display = isVisible ? 'none' : 'block';
                            heading.querySelector('i').className = `bi bi-chevron-${isVisible ? 'down' : 'up'}`;
                        }
                    };
                }
            });

            // Add tab navigation links
            responseElement.querySelectorAll('a[data-section]').forEach(link => {
                link.classList.add('dashboard-link');
                link.addEventListener('click', dashboardLinkClickHandler);
            });

            // Add tooltips for data points
            responseElement.querySelectorAll('.data-point').forEach(point => {
                const tooltip = document.createElement('div');
                tooltip.className = 'data-tooltip';
                tooltip.textContent = point.getAttribute('data-info');
                point.appendChild(tooltip);
            });
        }

        // Handle visualization suggestions and metrics
        if (data.visualization_suggestions?.length) displayVisualizationSuggestions(data.visualization_suggestions);
        if (data.metrics) displayRelatedMetrics(data.metrics);
    })
    .catch(error => {
        hideTypingIndicator();
        console.error('Error processing chat message:', error);
        addMessageToChat('assistant', `<div class='chatbot-response'><h3>Error</h3><p>I'm sorry, I couldn't process your request: ${error.message}</p></div>`);
    });
}

// Add a function to handle dashboard link clicks
function setupDashboardLinkListeners() {
    console.log("Setting up dashboard link listeners...");
    // Find all dashboard links in all assistant messages (not just the most recent)
    const dashboardLinks = document.querySelectorAll('.assistant-message .dashboard-link');
    
    console.log(`Found ${dashboardLinks.length} dashboard links`);
    
    dashboardLinks.forEach(link => {
        // Remove any existing click listeners to avoid duplicates
        link.removeEventListener('click', dashboardLinkClickHandler);
        link.addEventListener('click', dashboardLinkClickHandler);
    });
}

function dashboardLinkClickHandler(e) {
    e.preventDefault();
    console.log("Dashboard link clicked");
    
    // Get the target section from the data attribute
    const targetSection = this.getAttribute('data-section');
    console.log(`Target section: ${targetSection}`);
    
    // Map the section to the corresponding tab ID
    const tabMapping = {
        'timeseries': 'timeseries-tab',
        'network': 'network-tab',
        'topics': 'topics-tab',
        'coordinated': 'coordinated-tab',
        'word_cloud': 'word-cloud-tab',  // This doesn't exist, needs fixing
        'contributors': 'contributors-tab',
        'overview': 'overview-tab',
        'ai_insights': 'ai-insights-tab',
        'data_story': 'data-story-tab',
        'events': 'events-tab',
        'semantic_map': 'semantic-map-tab'
    };
    
    // Use proper tab IDs that actually exist in the document
    // The proper correction for tab IDs that don't exist in the HTML
    if (targetSection === 'word_cloud') {
        const overviewTab = document.getElementById('overview-tab');
        if (overviewTab) {
            overviewTab.click();
            console.log("Redirecting to overview tab (word cloud doesn't have its own tab)");
        } else {
            console.warn("Overview tab element not found");
        }
        return;
    }
    
    if (targetSection === 'ai_insights') {
        const overviewTab = document.getElementById('overview-tab');
        if (overviewTab) {
            overviewTab.click();
            console.log("Redirecting to overview tab (AI insights panel)");
        } else {
            console.warn("Overview tab element not found");
        }
        return;
    }
    
    if (targetSection === 'data_story') {
        const overviewTab = document.getElementById('overview-tab');
        if (overviewTab) {
            overviewTab.click();
            console.log("Redirecting to overview tab (data story panel)");
        } else {
            console.warn("Overview tab element not found");
        }
        return;
    }
    
    if (targetSection === 'events') {
        const timeseriesTab = document.getElementById('timeseries-tab');
        if (timeseriesTab) {
            timeseriesTab.click();
            console.log("Redirecting to timeseries tab (events are shown there)");
        } else {
            console.warn("Timeseries tab element not found");
        }
        return;
    }
    
    // For other tabs, use the mapping if available
    const tabElement = document.getElementById(tabMapping[targetSection]);
    if (tabElement) {
        console.log(`Clicking tab: ${tabMapping[targetSection]}`);
        tabElement.click();
        
        // Add a visual indication that the tab was navigated from chatbot
        tabElement.classList.add('highlight-tab');
        setTimeout(() => {
            tabElement.classList.remove('highlight-tab');
        }, 2000);
    } else {
        console.warn(`Tab for section "${targetSection}" not found`);
    }
}

function displayVisualizationSuggestions(suggestions) {
    const container = document.getElementById('viz-suggestions-content');
    if (!container) {
        console.warn('Visualization suggestions container not found');
        return;
    }
    
    container.innerHTML = '';
    
    suggestions.forEach(suggestion => {
        const col = document.createElement('div');
        col.className = 'col-md-6 col-lg-4';
        
        const card = document.createElement('div');
        card.className = 'card suggestion-card h-100';
        card.style.cursor = 'pointer';
        card.addEventListener('click', () => {
            // Switch to relevant tab based on suggestion type
            let tabElement;
            switch(suggestion.type) {
                case 'time_series':
                    tabElement = document.getElementById('timeseries-tab');
                    break;
                case 'topics':
                    tabElement = document.getElementById('topics-tab');
                    break;
                case 'network':
                    tabElement = document.getElementById('network-tab');
                    break;
                case 'semantic_map':
                    tabElement = document.getElementById('semantic-map-tab');
                    break;
                case 'community_distribution':
                    tabElement = document.getElementById('network-tab'); // This is on the network tab
                    break;
                case 'coordinated':
                    tabElement = document.getElementById('coordinated-tab');
                    break;
                default:
                    tabElement = document.getElementById('overview-tab');
                    break;
            }
            
            if (tabElement) {
                tabElement.click();
            } else {
                console.warn(`Tab for visualization type "${suggestion.type}" not found`);
            }
        });
        
        // Get appropriate icon for visualization type
        let icon = 'graph-up';
        switch(suggestion.type) {
            case 'time_series': icon = 'graph-up'; break;
            case 'topics': icon = 'chat-square-text'; break;
            case 'network': icon = 'diagram-3'; break;
            case 'semantic_map': icon = 'map'; break;
            case 'community_distribution': icon = 'pie-chart'; break;
            case 'coordinated': icon = 'people'; break;
            case 'overview': icon = 'list-ul'; break;
        }
        
        card.innerHTML = `
            <div class="card-body">
                <h5 class="card-title"><i class="bi bi-${icon}"></i> ${suggestion.title}</h5>
                <p class="card-text small">${suggestion.description}</p>
            </div>
            <div class="card-footer bg-transparent text-end">
                <small class="text-muted">Click to view</small>
            </div>
        `;
        
        col.appendChild(card);
        container.appendChild(col);
    });
    
    // Show the container
    const vizSuggestions = document.getElementById('viz-suggestions');
    if (vizSuggestions) {
        vizSuggestions.style.display = 'block';
    }
}

function displayRelatedMetrics(metrics) {
    const container = document.getElementById('metrics-content');
    if (!container) {
        console.warn('Metrics content container not found');
        return;
    }
    
    container.innerHTML = '';
    
    // Define the metrics to show and their formats
    const metricsToShow = [
        { key: 'total_posts', label: 'Total Posts', format: value => value.toLocaleString() },
        { key: 'unique_authors', label: 'Unique Authors', format: value => value.toLocaleString() },
        { key: 'avg_comments', label: 'Avg. Comments', format: value => value.toFixed(1) }
    ];
    
    // Add time range if available
    if (metrics.time_range) {
        const startDate = new Date(metrics.time_range.start).toLocaleDateString();
        const endDate = new Date(metrics.time_range.end).toLocaleDateString();
        
        const col = document.createElement('div');
        col.className = 'col-12';
        col.innerHTML = `
            <div class="alert alert-light">
                <small class="fw-medium">Data Range: ${startDate} to ${endDate}</small>
            </div>
        `;
        container.appendChild(col);
    }
    
    // Add standard metrics
    metricsToShow.forEach(metricDef => {
        if (metrics[metricDef.key] !== undefined) {
            const col = document.createElement('div');
            col.className = 'col-md-4';
            
            col.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${metricDef.format(metrics[metricDef.key])}</div>
                    <div class="stat-label">${metricDef.label}</div>
                </div>
            `;
            
            container.appendChild(col);
        }
    });
    
    // Add peak date if available (for trend analysis)
    if (metrics.peak_date) {
        const col = document.createElement('div');
        col.className = 'col-md-4';
        
        col.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${metrics.peak_date}</div>
                <div class="stat-label">Peak Activity</div>
            </div>
        `;
        
        container.appendChild(col);
    }
    
    // Add trend direction if available
    if (metrics.trend && metrics.trend.direction) {
        const col = document.createElement('div');
        col.className = 'col-md-4';
        
        const icon = metrics.trend.direction === 'increasing' ? 
            '<i class="bi bi-arrow-up-right text-success"></i>' : 
            '<i class="bi bi-arrow-down-right text-danger"></i>';
        
        col.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${icon} ${metrics.trend.direction.charAt(0).toUpperCase() + metrics.trend.direction.slice(1)}</div>
                <div class="stat-label">Overall Trend</div>
            </div>
        `;
        
        container.appendChild(col);
    }
    
    // Show top subreddits if available
    if (metrics.top_subreddits) {
        const col = document.createElement('div');
        col.className = 'col-12 mt-3';
        
        const subreddits = Object.entries(metrics.top_subreddits)
            .slice(0, 3)
            .map(([name, count]) => `<span class="badge bg-light text-dark">r/${name} (${count})</span>`)
            .join(' ');
        
        col.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Top Subreddits</h6>
                    <div>${subreddits}</div>
                </div>
            </div>
        `;
        
        container.appendChild(col);
    }
    
    // Show top keywords if available
    if (metrics.top_keywords && metrics.top_keywords.length > 0) {
        const col = document.createElement('div');
        col.className = 'col-12 mt-3';
        
        const keywords = metrics.top_keywords
            .slice(0, 5)
            .map(word => `<span class="badge bg-light text-dark">${word}</span>`)
            .join(' ');
        
        col.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <h6 class="card-title">Key Terms</h6>
                    <div>${keywords}</div>
                </div>
            </div>
        `;
        
        container.appendChild(col);
    }
    
    // Show the container
    const relatedMetrics = document.getElementById('related-metrics');
    if (relatedMetrics) {
        relatedMetrics.style.display = 'block';
    }
}
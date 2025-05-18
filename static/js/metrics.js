// Function to update key metrics
async function updateKeyMetrics(query) {
    try {
        const response = await fetch(`/api/metrics?query=${encodeURIComponent(query)}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const metrics = await response.json();
        
        // Update the metrics in the UI
        document.getElementById('total-posts').textContent = metrics.total_posts;
        document.getElementById('unique-authors').textContent = metrics.unique_authors;
        document.getElementById('avg-comments').textContent = metrics.avg_comments.toFixed(1);
        document.getElementById('time-span').textContent = metrics.time_span.toFixed(1);
    } catch (error) {
        console.error('Error updating key metrics:', error);
        // Show error state in the UI
        document.getElementById('total-posts').textContent = '-';
        document.getElementById('unique-authors').textContent = '-';
        document.getElementById('avg-comments').textContent = '-';
        document.getElementById('time-span').textContent = '-';
    }
}
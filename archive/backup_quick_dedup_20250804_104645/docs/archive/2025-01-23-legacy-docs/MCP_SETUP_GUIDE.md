# 🚀 Kimera MCP Servers Setup Guide

## Overview

This guide covers the complete setup of Model Context Protocol (MCP) servers for Cursor AI, including both native Python servers and Docker-based servers. The configuration provides enhanced cognitive capabilities aligned with Kimera's neurodivergent cognitive dynamics.

## 📋 Configured MCP Servers

### Native Python Servers (Already Working)
1. **kimera-simple** - Project-specific utilities
2. **fetch** - Web content retrieval
3. **sqlite-kimera** - Persistent cognitive memory

### New Docker-based Servers
4. **context7** - Advanced context management
5. **desktop-commander** - Desktop automation
6. **exa** - Enhanced web search
7. **memory** - Persistent memory storage
8. **neo4j-cypher** - Graph database queries
9. **neo4j-memory** - Graph-based memory
10. **node-code-sandbox** - Node.js code execution
11. **paper-search** - Academic paper research
12. **postgres** - PostgreSQL database access
13. **sequentialthinking** - Sequential reasoning chains

## 🔧 Prerequisites

### Required Software
- **Docker Desktop** (for Docker-based servers)
- **Python 3.10+** (for native servers)
- **Neo4j Database** (for graph-based servers)
- **PostgreSQL** (for postgres server)

### Environment Setup
1. **Install Docker Desktop**: Download from [docker.com](https://www.docker.com/products/docker-desktop/)
2. **Pull MCP Docker Images**:
   ```bash
   docker pull mcp/context7
   docker pull mcp/desktop-commander
   docker pull mcp/exa
   docker pull mcp/memory
   docker pull mcp/neo4j-cypher
   docker pull mcp/neo4j-memory
   docker pull mcp/node-code-sandbox
   docker pull mcp/paper-search
   docker pull mcp/postgres
   docker pull mcp/sequentialthinking
   ```

## 🔐 Security Configuration

### API Keys & Credentials
- **EXA_API_KEY**: `7c1258fb-75e9-40ce-9765-06d7a620e642` (configured)
- **Neo4j Credentials**: 
  - URL: `bolt://host.docker.internal:7687`
  - Username: `neo4j`
  - Password: `password`
- **PostgreSQL URL**: `postgresql://host.docker.internal:5432/mydb`

### Important Security Notes
⚠️ **The EXA API key is exposed in the configuration. For production use:**
1. Move sensitive credentials to environment variables
2. Use Docker secrets for production deployments
3. Rotate API keys regularly

## 🏗️ Configuration Details

### Volume Mounts
- **Memory Server**: Maps project directory to `/local-directory` for persistent storage
- **Path**: `D:\DEV\Kimera_SWM_Alpha_Prototype V0.1 140625:/local-directory`

### Docker Network Configuration
- **Host Access**: Uses `host.docker.internal` for database connections
- **Container Isolation**: Each server runs in isolated, ephemeral containers (`--rm`)
- **Interactive Mode**: All containers run with `-i` flag for JSON-RPC communication

## 🚀 Activation Steps

### 1. Verify Docker Setup
```bash
# Test Docker installation
docker --version
docker run hello-world

# Pull required images
docker pull mcp/context7
# ... (repeat for all servers)
```

### 2. Setup Database Services (Optional)
For Neo4j and PostgreSQL servers to work:

```bash
# Start Neo4j (if using graph databases)
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Start PostgreSQL (if using postgres server)
docker run -d --name postgres \
  -p 5432:5432 \
  -e POSTGRES_DB=mydb \
  -e POSTGRES_PASSWORD=password \
  postgres:latest
```

### 3. Restart Cursor AI
1. Close Cursor AI completely
2. Restart the application
3. Verify MCP servers are loaded in settings

## 🔍 Verification & Testing

### Check MCP Status
1. Open Cursor AI Settings
2. Navigate to "Model Context Protocol"
3. Verify all servers show as "Connected" or "Available"

### Test Server Functionality
Use these commands in Cursor's chat to test servers:

```bash
# Test native servers
- Ask for project status (kimera-simple)
- Request web content (fetch)
- Query database (sqlite-kimera)

# Test Docker servers
- Request context analysis (context7)
- Search academic papers (paper-search)
- Run Node.js code (node-code-sandbox)
```

## 🐛 Troubleshooting

### Common Issues

#### Docker Servers Not Starting
```bash
# Check Docker daemon
docker info

# Test container manually
docker run -i --rm mcp/context7
```

#### Path Issues on Windows
- Ensure paths use forward slashes in volume mounts
- Verify Docker Desktop has access to the drive
- Check Windows path escaping

#### Neo4j Connection Failures
```bash
# Verify Neo4j is running
docker ps | grep neo4j

# Test connection
docker exec -it neo4j cypher-shell -u neo4j -p password
```

#### Memory Server Volume Mount Issues
- Verify the project path exists
- Check Docker Desktop file sharing permissions
- Ensure no spaces in path cause issues

### Debug Mode
All servers are configured with `DEBUG: "*"` for verbose logging. Check Cursor's developer console for detailed error messages.

## 🧠 Cognitive Alignment

### Kimera Core Philosophy Integration
This MCP setup enhances Kimera's cognitive capabilities:

1. **Deep Context Sensitivity**: Multi-layered context through various servers
2. **Resonance-Triggered Exploration**: Web search and academic paper access
3. **Analogy as Core Bridge**: Graph databases for relationship mapping
4. **Multi-Perspectival Thinking**: Multiple data sources and reasoning chains
5. **Visual/Graphical Thinking**: Enhanced through structured data access

### Neurodivergent Cognitive Support
- **Working Memory Enhancement**: Persistent memory servers
- **Context Switching Support**: Multiple specialized servers
- **Pattern Recognition**: Graph-based and SQL-based analysis
- **Executive Function Support**: Sequential thinking and desktop automation

## 📊 Performance Expectations

### Server Load Times
- **Native Python**: ~1-2 seconds
- **Docker Servers**: ~3-5 seconds (first run)
- **Subsequent Requests**: <1 second

### Resource Usage
- **Memory**: ~50-100MB per Docker server
- **CPU**: Minimal during idle
- **Disk**: Ephemeral containers (minimal storage)

## 🔄 Maintenance

### Regular Tasks
1. **Update Docker Images**: `docker pull mcp/*` monthly
2. **Rotate API Keys**: Update EXA and other service keys
3. **Monitor Logs**: Check for connection issues
4. **Database Maintenance**: Backup Neo4j and PostgreSQL data

### Configuration Backup
The current configuration is automatically backed up in:
- `.cursor/mcp_backup.json`
- This documentation serves as configuration reference

## 🚨 Emergency Procedures

### Rollback Configuration
If new servers cause issues:
```bash
# Restore backup configuration
cp .cursor/mcp_backup.json .cursor/mcp.json
```

### Disable Problematic Servers
Comment out specific servers in `.cursor/mcp.json`:
```json
// "problematic-server": {
//   "command": "...",
//   ...
// }
```

## 📈 Success Metrics

### Expected Outcomes
- **Total MCP Tools**: 25+ (up from 9)
- **Enhanced Capabilities**: Web search, graph analysis, code execution
- **Cognitive Amplification**: Multi-modal reasoning support
- **Development Efficiency**: Automated tasks and enhanced context

### Monitoring
- Track tool usage patterns
- Monitor server response times
- Assess cognitive enhancement impact
- Document successful use cases

---

*This configuration represents a significant enhancement to Kimera's cognitive capabilities, providing a comprehensive toolkit for neurodivergent-friendly AI assistance.* 
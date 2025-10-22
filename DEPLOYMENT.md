# Streamlit Community Cloud Deployment Guide

## Option 1: Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is the easiest way to deploy your Streamlit app:

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Select your repository**: `MadiyarMoldabayev/E-NGO`
5. **Set the main file path**: `app.py`
6. **Add environment variables**:
   - `OPENAI_API_KEY`: Your OpenAI API key
7. **Click "Deploy!"**

Your app will be available at: `https://your-app-name.streamlit.app`

## Option 2: Heroku Deployment

If you prefer Heroku:

1. **Install Heroku CLI**
2. **Login to Heroku**:
   ```bash
   heroku login
   ```
3. **Create Heroku app**:
   ```bash
   heroku create your-app-name
   ```
4. **Set environment variables**:
   ```bash
   heroku config:set OPENAI_API_KEY=your_openai_api_key
   ```
5. **Deploy**:
   ```bash
   git push heroku main
   ```

## Option 3: Railway Deployment

Railway is another good option for Python apps:

1. **Go to**: https://railway.app/
2. **Connect your GitHub account**
3. **Select your repository**
4. **Add environment variables**
5. **Deploy automatically**

## Important Notes

- **Vector Store Files**: The large vector store files are not included in the repository. You'll need to rebuild them on the deployment platform or use a cloud storage service.
- **Environment Variables**: Make sure to set your `OPENAI_API_KEY` in the deployment platform.
- **Dependencies**: The `requirements.txt` file includes all necessary dependencies.

## Rebuilding Vector Store on Deployment

If you need to rebuild the vector store files on deployment, you can:

1. **Add the PDF files** to your repository (if they're not too large)
2. **Run the build script** during deployment:
   ```bash
   python build_indexes.py
   ```

Or use a cloud storage service like AWS S3 to store the vector files.

# Project RAG Pro — Cloud Deployment Guide (Render)

This guide will walk you through deploying your Project RAG Pro application to the cloud using Render. This will give you a public URL that your colleagues can access from anywhere.

**Time required:** ~10 minutes

---

### Step 1: Create a GitHub Account

If you don't have one, sign up for a free account at [github.com](https://github.com).

### Step 2: Create a New GitHub Repository

1.  Click the **+** icon in the top right and select **New repository**.
2.  Give it a name (e.g., `project-rag-pro`).
3.  Make it **Private** (recommended, to keep your code and data secure).
4.  Click **Create repository**.

### Step 3: Upload the Application Files

1.  In your new GitHub repository, click **Add file** > **Upload files**.
2.  Drag and drop all the files from the `project_rag_pro_deployment.zip` into the upload area:
    *   `app.py`
    *   `templates/index.html`
    *   `Dockerfile`
    *   `requirements.txt`
    *   `render.yaml`
    *   `.dockerignore`
    *   `.gitignore`
3.  Click **Commit changes**.

### Step 4: Create a Render Account

1.  Go to [dashboard.render.com/register](https://dashboard.render.com/register) and sign up using your GitHub account.
2.  Authorize Render to access your GitHub repositories.

### Step 5: Deploy the Application

1.  On the Render dashboard, click **New +** > **Blueprint**.
2.  Select your `project-rag-pro` repository from the list and click **Connect**.
3.  Render will automatically read the `render.yaml` file and configure everything for you. You just need to fill in the secrets.
4.  Scroll down to the **Environment** section and add two secrets:

    | Key | Value |
    | --- | --- |
    | `OPENAI_API_KEY` | Your `sk-...` key |
    | `APP_PASSWORD` | A password for your team (e.g., `MyTeamRAG2026!`) |

5.  Click **Apply**.

Render will now start building and deploying your application. This will take about 5-10 minutes the first time.

### Step 6: Access Your App

Once the deployment is complete, Render will give you a public URL at the top of the page, like `https://project-rag-pro.onrender.com`.

Share this URL and the password you set with your colleagues. They can now access the app from anywhere.

---

### Important Notes

*   **Free Tier:** The free plan on Render will "sleep" the app after 15 minutes of inactivity. The next person to visit will have to wait ~30 seconds for it to wake up. For always-on access, you can upgrade to the "Starter" plan for ~$7/month.
*   **Data Persistence:** The `render.yaml` file is configured to use a **Persistent Disk**. This means your uploaded documents and vector indexes are saved and will not be lost when the app restarts or sleeps.
*   **Updating the App:** If you want to update the app in the future, just push the changes to your GitHub repository. Render will automatically detect the changes and redeploy the new version.

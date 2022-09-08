# MLSA-Tutorial
Welcome to the MLSA Deep Learing Tutorial. In this session, we will help you install the necessary software and framework, perform Exploratory Data Analysis using **Power Bi**, design, train and test a Deep Learning Model using **Keras**, and deploy it using an intuitive UI using **Streamlit**.

> Date : <br>
> Time : <br>
> Mentors : <br>
> Venue :<br>
> Link :<br>

## Quick Start Guide

This guide will help you install the prerequisites to attend the MLSA Deep Learing Tutorial smoothly. Majority of this section is a self-task which shall not be covered during the session. If you have any doubts, feel free to contact any of the mentors for the session. Happy Deep Learning!

<details><summary><b>Show instructions</b></summary>

1. Install the preset:

    ```sh
    $ npm install --save-dev size-limit @size-limit/file
    ```

2. Add the `size-limit` section and the `size` script to your `package.json`:

    ```diff
    + "size-limit": [
    +   {
    +     "path": "dist/app-*.js"
    +   }
    + ],
      "scripts": {
        "build": "webpack ./webpack.config.js",
    +   "size": "npm run build && size-limit",
        "test": "jest && eslint ."
      }
    ```

3. Here’s how you can get the size for your current project:

    ```sh
    $ npm run size

      Package size: 30.08 kB with all dependencies, minified and gzipped
    ```

4. Now, let’s set the limit. Add 25% to the current total size and use that as
   the limit in your `package.json`:

    ```diff
      "size-limit": [
        {
    +     "limit": "35 kB",
          "path": "dist/app-*.js"
        }
      ],
    ```

5. Add the `size` script to your test suite:

    ```diff
      "scripts": {
        "build": "webpack ./webpack.config.js",
        "size": "npm run build && size-limit",
    -   "test": "jest && eslint ."
    +   "test": "jest && eslint . && npm run size"
      }
    ```

6. If you don’t have a continuous integration service running, don’t forget
   to add one — start with [Travis CI].

</details>

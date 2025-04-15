{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPbZvrMf8+TOIhilwJTHbDC",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Anissawen/Homework-MACS-30123/blob/main/Pset1_p1b_whole.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Q1. a"
      ],
      "metadata": {
        "id": "yXTHALihJDFr"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Step1: first we try to calculate the original code time"
      ],
      "metadata": {
        "id": "QtsD9qUkCbRp"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "w_fi_Cx478a-",
        "outputId": "bd539d0a-706e-45f1-85f6-f1bd120c2429"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Elapsed time: 16.3990 seconds\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "import scipy.stats as sts\n",
        "import time\n",
        "\n",
        "# Set model parameters\n",
        "rho = 0.5\n",
        "mu = 3.0\n",
        "sigma = 1.0\n",
        "z_0 = mu\n",
        "\n",
        "# Set simulation parameters\n",
        "S = 1000  # Number of lives to simulate\n",
        "T = 4160  # Number of periods for each simulation\n",
        "np.random.seed(25)\n",
        "\n",
        "# Draw all idiosyncratic random shocks and create containers\n",
        "eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))\n",
        "z_mat = np.zeros((T, S))\n",
        "\n",
        "start_time = time.time()\n",
        "for s_ind in range(S):\n",
        "    z_tm1 = z_0\n",
        "    for t_ind in range(T):\n",
        "        e_t = eps_mat[t_ind, s_ind]\n",
        "        z_t = rho * z_tm1 + (1 - rho) * mu + e_t\n",
        "        z_mat[t_ind, s_ind] = z_t\n",
        "        z_tm1 = z_t\n",
        "end_time = time.time()\n",
        "print(f\"Elapsed time: {end_time - start_time:.4f} seconds\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Step2: we use numba.cc to precompile our codes"
      ],
      "metadata": {
        "id": "7YQ-UE78DT2O"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from numba.pycc import CC\n",
        "import numpy as np\n",
        "\n",
        "cc = CC(\"q1a\")\n",
        "\n",
        "# Export function: x is 2D float64 (f8) arrays, and scalar inputs are int32 (i4) or float64 (f8)\n",
        "@cc.export('simulate_lifetimes', 'void(f8[:,:], f8[:,:], f8, f8, f8, i4, i4)')\n",
        "def simulate_lifetimes(z_mat, eps_mat, rho, mu, sigma, S, T):\n",
        "    for s_ind in range(S):\n",
        "        z_tm1 = mu\n",
        "        for t_ind in range(T):\n",
        "            e_t = eps_mat[t_ind, s_ind]\n",
        "            z_t = rho * z_tm1 + (1 - rho) * mu + e_t\n",
        "            z_mat[t_ind, s_ind] = z_t\n",
        "            z_tm1 = z_t\n",
        "\n",
        "cc.compile()\n"
      ],
      "metadata": {
        "id": "Q3Sf1sFoCX3U"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import scipy.stats as sts\n",
        "import time\n",
        "import q1a  # the precompiled module\n",
        "\n",
        "# Parameters\n",
        "rho = 0.5\n",
        "mu = 3.0\n",
        "sigma = 1.0\n",
        "z_0 = mu\n",
        "S = 1000  # Number of lifetimes\n",
        "T = 4160  # Number of weeks (80 years × 52)\n",
        "\n",
        "np.random.seed(25)\n",
        "eps_mat = np.random.normal(loc=0, scale=sigma, size=(T, S))\n",
        "z_mat = np.zeros((T, S))\n",
        "\n",
        "# Time performance\n",
        "start = time.time()\n",
        "q1a.simulate_lifetimes(z_mat, eps_mat, rho, mu, sigma, S, T)\n",
        "end = time.time()\n",
        "\n",
        "print(f\"AOT-compiled simulation took: {end - start:.4f} seconds\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HZ6_lBiFDNsO",
        "outputId": "dee39a63-945e-47ea-a3f1-0bc93c78540f"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "AOT-compiled simulation took: 0.0712 seconds\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Q1.b"
      ],
      "metadata": {
        "id": "TJFp3TahJB6w"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import scipy.stats as sts\n",
        "import time\n",
        "import q1a  # the precompiled module\n",
        "from mpi4py import MPI\n",
        "\n",
        "# Initialize MPI\n",
        "comm = MPI.COMM_WORLD\n",
        "rank = comm.Get_rank()\n",
        "\n",
        "# Set model parameters\n",
        "rho = 0.5\n",
        "mu = 3.0\n",
        "sigma = 1.0\n",
        "z_0 = mu\n",
        "\n",
        "# Set simulation parameters\n",
        "S = 1000       # number of lives to simulate\n",
        "T = 4160       # number of periods for each simulation\n",
        "np.random.seed(rank)\n",
        "\n",
        "# Draw all shocks\n",
        "eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S)).astype(np.float64)\n",
        "\n",
        "# Run and time the simulation\n",
        "start_time = time.time()\n",
        "z_mat = q1a.simulate_lifetimes(z_mat, eps_mat, rho, mu, sigma, S, T)\n",
        "end_time = time.time()\n",
        "elapsed = end_time - start_time\n",
        "\n",
        "# Only print the elapsed time from rank 0\n",
        "if rank == 0:\n",
        "    print(f\"Elapsed time with ahead-of-time compiled function: {elapsed:.4f} seconds\")"
      ],
      "metadata": {
        "id": "hqNBDgaDdknw"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
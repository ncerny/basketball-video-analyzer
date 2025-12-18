import axios, { AxiosError } from 'axios'
import { APIError, type ErrorResponse } from '../types/api'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
})

// Request interceptor for logging and auth
apiClient.interceptors.request.use(
  (config) => {
    // Log requests in development
    if (import.meta.env.DEV) {
      console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`)
    }

    // Add auth token when implemented
    // const token = getAuthToken()
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`
    // }

    return config
  },
  (error) => {
    console.error('[API] Request error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    // Log successful responses in development
    if (import.meta.env.DEV) {
      console.log(`[API] ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status}`)
    }
    return response
  },
  (error: AxiosError<ErrorResponse>) => {
    // Handle network errors
    if (!error.response) {
      const networkError = new APIError(
        'Network error. Please check your connection.',
        undefined,
        'NETWORK_ERROR'
      )
      console.error('[API] Network error:', error.message)
      return Promise.reject(networkError)
    }

    // Handle API errors
    const status = error.response.status
    const data = error.response.data

    let message = data?.message || 'An unexpected error occurred'
    const code = data?.code

    // Customize error messages based on status
    switch (status) {
      case 400:
        message = data?.message || 'Bad request'
        break
      case 401:
        message = 'Unauthorized. Please log in.'
        break
      case 403:
        message = 'Forbidden. You do not have permission.'
        break
      case 404:
        message = data?.message || 'Resource not found'
        break
      case 500:
        message = 'Server error. Please try again later.'
        break
      case 503:
        message = 'Service unavailable. Please try again later.'
        break
    }

    const apiError = new APIError(message, status, code, data?.details)
    console.error(`[API] Error ${status}:`, message)

    return Promise.reject(apiError)
  }
)

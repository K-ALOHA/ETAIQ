import { render, screen, waitFor } from '@testing-library/react';
import DashboardPage from './page';

const mockFetch = vi.fn();

global.fetch = mockFetch as unknown as typeof fetch;

describe('DashboardPage', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('loads the dashboard from model and monitoring endpoints without invoking prediction', async () => {
    mockFetch.mockImplementation((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;

      if (url.includes('/api/v1/models')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            current_model: 'XGBRegressor_v2',
            version: 2,
            created_at: '2026-01-01T00:00:00Z',
            available_models: ['XGBRegressor_v2'],
            models: [
              {
                model_name: 'XGBRegressor',
                version: 2,
                artifact_path: '/models/XGBRegressor_v2.joblib',
                status: 'Production',
                metrics: { mae: 1.23, rmse: 2.34, r2: 0.87 },
                dataset_size: 1200,
                training_samples: 1000,
              },
            ],
            count: 1,
          }),
        }) as Promise<Response>;
      }

      if (url.includes('/api/v1/monitoring')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ records: [], count: 0 }),
        }) as Promise<Response>;
      }

      if (url.includes('/api/v1/models/registry')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ models: [], count: 0 }),
        }) as Promise<Response>;
      }

      if (url.includes('/api/v1/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'healthy', model_loaded: true }),
        }) as Promise<Response>;
      }

      return Promise.reject(new Error(`Unexpected request: ${url}`));
    });

    render(<DashboardPage />);

    expect(await screen.findByText(/Production dashboard/i)).toBeInTheDocument();

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(4));

    const predictionCalls = mockFetch.mock.calls.filter(([input, init]) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
      return url.includes('/api/v1/predict') || init?.method === 'POST';
    });

    expect(predictionCalls).toHaveLength(0);
  });
});

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PredictionPage from './page';

const mockFetch = vi.fn();
const mockPush = vi.fn();

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}));

global.fetch = mockFetch as unknown as typeof fetch;

describe('PredictionPage', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockPush.mockReset();
  });

  it('renders the production model summary and validates required fields', async () => {
    mockFetch
      .mockResolvedValueOnce({
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
              feature_names: ['drop_lat', 'drop_lon', 'order_size'],
              feature_count: 3,
            },
          ],
          count: 1,
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ prediction: 18.5, model_name: 'XGBRegressor', model_version: 2, processing_time_ms: 45.2 }),
      });

    render(<PredictionPage />);

    expect(await screen.findByText(/^Production model$/i)).toBeInTheDocument();
    expect(screen.getAllByText(/XGBRegressor/i).length).toBeGreaterThan(0);

    const submit = screen.getByRole('button', { name: /predict eta/i });
    await userEvent.click(submit);

    expect(await screen.findByText(/Drop Lat is required/i)).toBeInTheDocument();

    const dropLatInput = screen.getByLabelText(/Drop Lat/i);
    const dropLonInput = screen.getByLabelText(/Drop Lon/i);
    const orderSizeInput = screen.getByLabelText(/Order Size/i);

    await userEvent.clear(dropLatInput);
    await userEvent.type(dropLatInput, '3.4');
    await userEvent.clear(dropLonInput);
    await userEvent.type(dropLonInput, '4.2');
    await userEvent.clear(orderSizeInput);
    await userEvent.type(orderSizeInput, '7.1');
    await userEvent.click(submit);

    await waitFor(() => expect(mockFetch).toHaveBeenCalledTimes(3));
    expect(await screen.findByText(/Predicted ETA/i)).toBeInTheDocument();
  });

  it('shows a friendly error when the prediction API is unavailable', async () => {
    mockFetch
      .mockResolvedValueOnce({
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
              feature_names: ['drop_lat'],
              feature_count: 1,
            },
          ],
          count: 1,
        }),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        text: async () => 'Service unavailable',
        json: async () => ({ detail: 'Service unavailable' }),
      });

    render(<PredictionPage />);

    const input = await screen.findByLabelText(/Drop Lat/i);
    await userEvent.type(input, '3.4');
    await userEvent.click(screen.getByRole('button', { name: /predict eta/i }));

    expect(await screen.findByText(/Service unavailable/i)).toBeInTheDocument();
  });

  it('renders dynamic feature controls and surfaces explainability navigation', async () => {
    mockFetch
      .mockResolvedValueOnce({
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
              feature_names: ['drop_lat', 'order_size', 'promo_code_used', 'is_priority'],
              feature_count: 4,
              metrics: { mae: 0.8, rmse: 1.2 },
              feature_types: {
                drop_lat: 'numeric',
                order_size: 'numeric',
                promo_code_used: 'categorical',
                is_priority: 'boolean',
              },
            },
          ],
          count: 1,
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ prediction: 18.5, model_name: 'XGBRegressor', model_version: 2, processing_time_ms: 45.2 }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          confidence_score: 0.91,
          ranked_features: [
            { feature: 'drop_lat', contribution: 0.41 },
            { feature: 'order_size', contribution: 0.29 },
          ],
        }),
      });

    render(<PredictionPage />);

    expect(await screen.findByText(/Feature count/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/drop lat/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/promo code used/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/is priority/i)).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText(/drop lat/i), '3.4');
    await userEvent.type(screen.getByLabelText(/order size/i), '7.1');
    await userEvent.selectOptions(screen.getByLabelText(/promo code used/i), 'yes');
    await userEvent.click(screen.getByLabelText(/is priority/i));
    await userEvent.click(screen.getByRole('button', { name: /predict eta/i }));

    expect(await screen.findByText(/Predicted ETA/i)).toBeInTheDocument();
    expect(await screen.findByText(/0.91/i)).toBeInTheDocument();

    const explainButton = await screen.findByRole('button', { name: /view full explainability/i });
    await userEvent.click(explainButton);

    expect(mockPush).toHaveBeenCalledWith('/explainability');
  });
});

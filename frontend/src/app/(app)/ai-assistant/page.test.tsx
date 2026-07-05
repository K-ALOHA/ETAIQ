import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AIAssistantPage from './page';

const mockFetch = vi.fn();

global.fetch = mockFetch as unknown as typeof fetch;

describe('AIAssistantPage', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('displays welcome panel with suggested prompts initially', async () => {
    render(<AIAssistantPage />);

    expect(screen.getByText('ETAIQ AI Assistant')).toBeInTheDocument();
    expect(screen.getByText('What production model is running?')).toBeInTheDocument();
    expect(screen.getByText('Show model performance.')).toBeInTheDocument();
    expect(screen.getByText('Summarize the dataset.')).toBeInTheDocument();
    expect(screen.getByText('Why is the ETA high?')).toBeInTheDocument();
    expect(screen.getByText('Predict ETA.')).toBeInTheDocument();
  });

  it('sends message and displays response', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        response: 'This is a test response from the assistant!',
        conversation_id: 'test-convo-123',
        sources: [],
      }),
    } as Response);

    render(<AIAssistantPage />);

    // Click a suggested prompt
    fireEvent.click(screen.getByText('What production model is running?'));

    // Check that user message is displayed
    expect(screen.getByText('What production model is running?')).toBeInTheDocument();
    // Check loading state
    expect(screen.getByText('ETAIQ Assistant is thinking…')).toBeInTheDocument();

    // Wait for response
    await waitFor(() => {
      expect(screen.getByText('This is a test response from the assistant!')).toBeInTheDocument();
    });

    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it('displays error gracefully', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'));

    render(<AIAssistantPage />);

    // Send a message
    const textarea = screen.getByPlaceholderText('Ask ETAIQ anything...');
    fireEvent.change(textarea, { target: { value: 'Test error' } });
    fireEvent.click(screen.getByText('Send'));

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });

  it('displays loading state when waiting for response', async () => {
    mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves

    render(<AIAssistantPage />);

    const textarea = screen.getByPlaceholderText('Ask ETAIQ anything...');
    fireEvent.change(textarea, { target: { value: 'Test loading' } });
    fireEvent.click(screen.getByText('Send'));

    await waitFor(() => {
      expect(screen.getByText('ETAIQ Assistant is thinking…')).toBeInTheDocument();
    });
  });
});

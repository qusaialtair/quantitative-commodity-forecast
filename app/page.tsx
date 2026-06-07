import Dashboard from "@/components/Dashboard";
import { MOCK_DASHBOARD } from "@/lib/mock-data";

export default function Page() {
  return <Dashboard initial={MOCK_DASHBOARD} />;
}

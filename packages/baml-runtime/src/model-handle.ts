export interface ModelCapabilities {
	tools: boolean;
	images: boolean;
	streaming: boolean;
	thinking?: boolean;
}

export interface ModelHandle {
	id: string;
	displayName: string;
	family?: string;
	capabilities: ModelCapabilities;
	raw: unknown;
}
